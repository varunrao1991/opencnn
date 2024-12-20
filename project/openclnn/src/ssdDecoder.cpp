#include "ssdDecoder.h"
#include "bmpreader.h"

#include "numpy.hpp"

#include <array>
#include <fstream>
#include <numeric>
#include <random>
#include <string>

const uint32_t kNumberOfBoxParameters = 4;
const uint32_t kChannelCount = 3;

std::vector<SSDDecoder::RGB> SSDDecoder::GenerateRGBValues(uint32_t totalClasses)
{
    std::vector<RGB> rgbValues;
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution dis(0, 255);

    for (uint32_t i = 0; i < totalClasses; ++i)
    {
        RGB rgb;
        rgb.red = dis(gen);
        rgb.green = dis(gen);
        rgb.blue = dis(gen);
        rgbValues.push_back(rgb);
    }

    return rgbValues;
}

SSDDecoder::SSDDecoder(const std::string &name, std::shared_ptr<OpenclWrapper> openclWrapper) :
    Layer(name, openclWrapper, 2)
{
}

static const std::vector<std::string> s_kKernelNames = {};
const std::vector<std::string> &SSDDecoder::GetKernelNames()
{
    return s_kKernelNames;
}

void SSDDecoder::SetParameters(const std::vector<std::string> &elements)
{
    if (elements.size() <= 3)
    {
        ALOG_GPUML("SSDDecoder element size must be greater than 3");
        return;
    }
    else
    {
        std::vector<uint32_t> parameters(elements.size());
        uint32_t numCount = 0;
        for (uint32_t i = 0; i < elements.size(); i++)
        {
            parameters[numCount++] = std::atoi(elements[i].c_str());
        }
        m_totalClasses = parameters[2];
        m_width = parameters[0];
        m_height = parameters[1];

        m_inputSize.resize(elements.size() - 3);
        std::copy(parameters.begin() + 3, parameters.end(), m_inputSize.begin());

        m_totalBoxes = std::accumulate(m_inputSize.begin(), m_inputSize.end(), 0, std::plus<uint32_t>());

        m_boxes = std::vector<float>(m_totalBoxes * kNumberOfBoxParameters, 0);
        m_boxesDeltas = std::vector<float>(m_totalBoxes * kNumberOfBoxParameters, 0);
        m_labels = std::vector<float>(m_totalBoxes * m_totalClasses, 0);
        m_scores = std::vector<float>(m_totalBoxes, 0);
        m_categories = std::vector<uint8_t>(m_totalBoxes, 0);
        m_filteredBoxes = std::vector<Box>(m_totalBoxes);
        m_result = std::vector<Box>(m_totalBoxes);
        m_keep = std::vector<bool>(m_totalBoxes, 0);
        m_indices = std::vector<size_t>(m_totalBoxes, 0);
        m_colors = GenerateRGBValues(m_totalClasses);
    }
}

void SSDDecoder::EnqueueKernel()
{
    uint32_t startIndex = 0;
    for (size_t featureMapIndex = 0; featureMapIndex < m_inputSize.size(); featureMapIndex++)
    {
        const uint32_t kFeatureSize = kNumberOfBoxParameters * m_inputSize[featureMapIndex];

        clEnqueueReadBuffer(m_openclWrapper->m_commandQueue,
            m_src[featureMapIndex]->GetBuffer(),
            CL_TRUE,
            0,
            kFeatureSize * sizeof(float),
            m_boxesDeltas.data() + startIndex,
            0,
            0,
            0);
        startIndex += kFeatureSize;
    }
    startIndex = 0;
    for (size_t featureMapIndex = 0; featureMapIndex < m_inputSize.size(); featureMapIndex++)
    {
        const uint32_t kFeatureSize = m_totalClasses * m_inputSize[featureMapIndex];
        clEnqueueReadBuffer(m_openclWrapper->m_commandQueue,
            m_src[featureMapIndex + m_inputSize.size()]->GetBuffer(),
            CL_TRUE,
            0,
            kFeatureSize * sizeof(float),
            m_labels.data() + startIndex,
            0,
            0,
            0);
        startIndex += kFeatureSize;
    }
    DeltaToBoxes();
    ignoreBackground(true);
}

void SSDDecoder::ignoreBackground(bool enable)
{
    for (size_t boxIndex = 0; boxIndex < m_totalBoxes; boxIndex++)
    {
        auto maxValueIterator = std::max_element(
            m_labels.begin() + boxIndex * m_totalClasses, m_labels.begin() + (boxIndex + 1) * m_totalClasses);
        auto location = std::distance(m_labels.begin() + boxIndex * m_totalClasses, maxValueIterator);
        if (location == 0 && enable)
        {
            *maxValueIterator = 0;
        }
        m_categories[boxIndex] = location;
        m_scores[boxIndex] = *maxValueIterator;
    }
}

// Function to calculate intersection over union (IoU) between two boxes
float CalculateIoU(const Box &box1, const Box &box2)
{
    float area1 = (box1.x2 - box1.x1 + 1) * (box1.y2 - box1.y1 + 1);
    float area2 = (box2.x2 - box2.x1 + 1) * (box2.y2 - box2.y1 + 1);

    float xA = std::max(box1.x1, box2.x1);
    float yA = std::max(box1.y1, box2.y1);
    float xB = std::min(box1.x2, box2.x2);
    float yB = std::min(box1.y2, box2.y2);

    float intersectionArea = std::max(0.0f, xB - xA + 1) * std::max(0.0f, yB - yA + 1);
    float unionArea = area1 + area2 - intersectionArea;

    return intersectionArea / unionArea;
}

// Function to perform non-maximum suppression
void SSDDecoder::PerformNMS(const std::vector<float> &boxes,
    const std::vector<float> &scores,
    const std::vector<uint8_t> &categories,
    float iouThreshold,
    float scoreThreshold,
    uint16_t maxPossibleBoxes)
{
    m_detectedCount = scores.size();

    // Create Box objects and store their indices
    for (size_t i = 0; i < scores.size(); ++i)
    {
        m_result[i] = { boxes[kNumberOfBoxParameters * i],
            boxes[kNumberOfBoxParameters * i + 1],
            boxes[kNumberOfBoxParameters * i + 2],
            boxes[kNumberOfBoxParameters * i + 3],
            scores[i],
            categories[i] };
        m_indices[i] = i;
        m_keep[i] = true;
    }

    // Sort the boxes based on scores in descending order
    std::sort(m_indices.begin(),
        m_indices.end(),
        [&](size_t i1, size_t i2) { return m_result[i1].score > m_result[i2].score; });

    // Perform non-maximum suppression
    for (size_t i = 0; i < std::min(static_cast<size_t>(maxPossibleBoxes), m_result.size() - 1); ++i)
    {
        auto &boxSelected = m_result[m_indices[i]];
        if (!m_keep[i])
            continue;

        for (size_t j = i + 1; j < m_result.size(); ++j)
        {
            if (!m_keep[j])
                continue;
            auto &boxToCompare = m_result[m_indices[j]];
            if (boxToCompare.category == boxSelected.category)
            {
                if (CalculateIoU(boxSelected, boxToCompare) > iouThreshold)
                {
                    m_keep[j] = false;
                }
            }
        }
    }

    uint32_t count = 0;
    for (size_t i = 0; i < m_result.size(); ++i)
    {
        auto &boxSelected = m_result[m_indices[i]];
        if (m_keep[i] && boxSelected.score > scoreThreshold)
        {
            m_filteredBoxes[count++] = boxSelected;
        }
    }
    m_detectedCount = count;
}

void SSDDecoder::DeltaToBoxes()
{
    for (size_t boxIndex = 0; boxIndex < m_totalBoxes; boxIndex++)
    {
        const auto v0 = m_boxesDeltas[boxIndex * kNumberOfBoxParameters];
        const auto v1 = m_boxesDeltas[boxIndex * kNumberOfBoxParameters + 1];
        const auto v2 = m_boxesDeltas[boxIndex * kNumberOfBoxParameters + 2];
        const auto v3 = m_boxesDeltas[boxIndex * kNumberOfBoxParameters + 3];

        auto a0 = m_priorBoxes[boxIndex * kNumberOfBoxParameters];
        auto a1 = m_priorBoxes[boxIndex * kNumberOfBoxParameters + 1];
        auto a2 = m_priorBoxes[boxIndex * kNumberOfBoxParameters + 2];
        auto a3 = m_priorBoxes[boxIndex * kNumberOfBoxParameters + 3];

        auto pBoxWidth = a3 - a1;
        auto pBoxHeight = a2 - a0;
        auto pBoxX = a1 + 0.5f * pBoxWidth;
        auto pBoxY = a0 + 0.5f * pBoxHeight;

        auto d0 = v0 * m_variances[0];
        auto d1 = v1 * m_variances[1];
        auto d2 = v2 * m_variances[2];
        auto d3 = v3 * m_variances[3];

        auto aBoxWidth = exp(d3) * pBoxWidth;
        auto aBoxHeight = exp(d2) * pBoxHeight;
        auto aBoxX = d1 * pBoxWidth + pBoxX;
        auto aBoxY = d0 * pBoxHeight + pBoxY;

        auto x1 = aBoxX - (0.5 * aBoxWidth);
        auto y1 = aBoxY - (0.5 * aBoxHeight);
        auto x2 = aBoxWidth + x1;
        auto y2 = aBoxHeight + y1;

        m_boxes[boxIndex * kNumberOfBoxParameters] = x1;
        m_boxes[boxIndex * kNumberOfBoxParameters + 1] = y1;
        m_boxes[boxIndex * kNumberOfBoxParameters + 2] = x2;
        m_boxes[boxIndex * kNumberOfBoxParameters + 3] = y2;
    }
}

void SSDDecoder::SetKernelArguments() {}

void SSDDecoder::DrawBox(uint8_t *outBuffer,
    float x1,
    float y1,
    float x2,
    float y2,
    uint32_t width1,
    uint32_t height1,
    const SSDDecoder::RGB &rgbValue)
{
    auto width = static_cast<int32_t>(width1);
    auto height = static_cast<int32_t>(height1);

    uint32_t xAbsolute1 =
        static_cast<uint32_t>(std::min(width - 1, std::max(0, static_cast<int32_t>(x1 * static_cast<float>(width)))));
    uint32_t xAbsolute2 =
        static_cast<uint32_t>(std::min(width - 1, std::max(0, static_cast<int32_t>(x2 * static_cast<float>(width)))));
    uint32_t yAbsolute1 =
        static_cast<uint32_t>(std::min(height - 1, std::max(0, static_cast<int32_t>(y1 * static_cast<float>(height)))));
    uint32_t yAbsolute2 =
        static_cast<uint32_t>(std::min(height - 1, std::max(0, static_cast<int32_t>(y2 * static_cast<float>(height)))));

    for (uint32_t x = xAbsolute1; x < xAbsolute2; x++)
    {
        outBuffer[3 * x * height + 3 * yAbsolute1 + 0] = rgbValue.red;
        outBuffer[3 * x * height + 3 * yAbsolute1 + 1] = rgbValue.green;
        outBuffer[3 * x * height + 3 * yAbsolute1 + 2] = rgbValue.blue;

        outBuffer[3 * x * height + 3 * yAbsolute2 + 0] = rgbValue.red;
        outBuffer[3 * x * height + 3 * yAbsolute2 + 1] = rgbValue.green;
        outBuffer[3 * x * height + 3 * yAbsolute2 + 2] = rgbValue.blue;
    }

    for (uint32_t y = yAbsolute1; y < yAbsolute2; y++)
    {
        outBuffer[3 * xAbsolute1 * height + 3 * y + 0] = rgbValue.red;
        outBuffer[3 * xAbsolute1 * height + 3 * y + 1] = rgbValue.green;
        outBuffer[3 * xAbsolute1 * height + 3 * y + 2] = rgbValue.blue;

        outBuffer[3 * xAbsolute2 * height + 3 * y + 0] = rgbValue.red;
        outBuffer[3 * xAbsolute2 * height + 3 * y + 1] = rgbValue.green;
        outBuffer[3 * xAbsolute2 * height + 3 * y + 2] = rgbValue.blue;
    }
}

void SSDDecoder::GetDimension(std::vector<std::vector<uint32_t>> &dimension)
{
    dimension = { { m_width, m_height, 3 } };
}

void SSDDecoder::CopyOutputBuffer(uint8_t *outBuffer, int32_t)
{
#if 1
    const auto kWhiteColor = SSDDecoder::RGB{ 255, 255, 255 };
    //DrawBox(outBuffer, 0.1, 0.1, 0.9, 0.9, m_width, m_height, kWhiteColor);
    PerformNMS(m_boxes, m_scores, m_categories, 0.1f, 0.3f);
    for (size_t boxIndex = 0; boxIndex < m_detectedCount; boxIndex++)
    {
        const auto &box = m_filteredBoxes[boxIndex];
        auto colorRGB = box.category < m_colors.size() ? m_colors[box.category] : SSDDecoder::RGB{ 255, 255, 255 };
        DrawBox(outBuffer, box.x1, box.y1, box.x2, box.y2, m_width, m_height, colorRGB);
    }
#else
    auto scoreThreshold = 0.01;
    for (size_t boxIndex = 0; boxIndex < m_totalBoxes; boxIndex++)
    {
        auto probabilityOfInterest = m_scores[boxIndex];
        auto labelIndex = m_categories[boxIndex];
        if (probabilityOfInterest > scoreThreshold)
        {
            ALOG_GPUML("Label index is %d", labelIndex);
            auto colorRGB = labelIndex < m_colors.size() ? m_colors[labelIndex] : SSDDecoder::RGB{ 255, 255, 255 };
            auto v0 = m_boxes[boxIndex * kNumberOfBoxParameters];
            auto v1 = m_boxes[boxIndex * kNumberOfBoxParameters + 1];
            auto v2 = m_boxes[boxIndex * kNumberOfBoxParameters + 2];
            auto v3 = m_boxes[boxIndex * kNumberOfBoxParameters + 3];
            DrawBox(outBuffer, v0, v1, v2, v3, width, height, colorRGB);
        }
    }
#endif
}

void SSDDecoder::CreateBuffers(const std::vector<std::shared_ptr<DataContainerOpenCLFloat>> &src)
{
    m_src = src;
}

void SSDDecoder::FillLayerConstants(const std::filesystem::path &inputPath)
{
    std::vector<int> priorBoxShape;
    auto fullPath = inputPath / (m_name + "_prior_boxes.npy");

    aoba::LoadArrayFromNumpy(fullPath.string(), priorBoxShape, m_priorBoxes);

    fullPath = inputPath / (m_name + "_variances.npy");
    std::vector<int> varianceShape;

    aoba::LoadArrayFromNumpy(fullPath.string(), varianceShape, m_variances);
}

void SSDDecoder::DisplayOutputBuffer()
{
    uint32_t h = m_totalBoxes;
    uint32_t w = kNumberOfBoxParameters;

    for (uint32_t j = 0; j < std::min(10u, h); j++)
    {
        for (uint32_t i = 0; i < std::min(10u, w); i++)
            ALOG_GPUML_NO_NEWLINE("%5.6f\t", m_boxesDeltas[i * w + j]);
        ALOG_GPUML_NO_NEWLINE("\n");
    }
    ALOG_GPUML_NO_NEWLINE("\n");

    w = m_totalClasses;

    for (uint32_t j = 0; j < std::min(10u, h); j++)
    {
        for (uint32_t i = 0; i < std::min(10u, w); i++)
            ALOG_GPUML_NO_NEWLINE("%5.6f\t", m_labels[i * w + j]);
        ALOG_GPUML_NO_NEWLINE("\n");
    }
    ALOG_GPUML_NO_NEWLINE("\n");
}

SSDDecoder::~SSDDecoder() {}
