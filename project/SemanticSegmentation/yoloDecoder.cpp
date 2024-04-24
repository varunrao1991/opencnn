#include "yoloDecoder.h"
#include "bmpreader.h"

#include "numpy.hpp"

#include <array>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <random>
#include <string>

const uint32_t kNumberOfBoxParameters = 4;
const uint32_t kAnchorDimension = 2;
const uint32_t kChannelCount = 3;

std::vector<YOLODecoder::RGB> YOLODecoder::GenerateRGBValues(uint32_t totalClasses)
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

YOLODecoder::YOLODecoder(const std::string &name, std::shared_ptr<OpenclWrapper> openclWrapper) :
    Layer(name, openclWrapper, 2),
    m_anchorCount{},
    m_detectedCount{},
    m_totalBoxes{},
    m_totalClasses{},
    m_width{},
    m_height{}
{
}

static const std::vector<std::string> s_kKernelNames = {};
const std::vector<std::string> &YOLODecoder::GetKernelNames()
{
    return s_kKernelNames;
}

void YOLODecoder::SetParameters(const std::vector<std::string> &elements)
{
    if (elements.size() <= 4)
    {
        ALOG_GPUML("YOLODecoder element size must be greater than 3");
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
        m_anchorCount = parameters[3];

        m_inputSize.resize(elements.size() - 4);
        std::copy(parameters.begin() + 4, parameters.end(), m_inputSize.begin());

        auto totalBoxes = 1;
        for (auto it = m_inputSize.begin(); it != m_inputSize.end(); it++)
        {
            totalBoxes *= *it * *it;
        }
        totalBoxes *= m_anchorCount;
        m_totalBoxes = totalBoxes;

        m_boxesDeltasScoreLabel = std::vector<float>(m_totalBoxes * (kNumberOfBoxParameters + m_totalClasses + 1), 0);
        m_boxes = std::vector<float>(m_totalBoxes * kNumberOfBoxParameters, 0);
        m_categories = std::vector<uint8_t>(m_totalBoxes, 0);
        m_filteredBoxes = std::vector<Box>(m_totalBoxes);
        m_scores = std::vector<float>(m_totalBoxes, 0);
        m_result = std::vector<Box>(m_totalBoxes);
        m_keep = std::vector<bool>(m_totalBoxes, 0);
        m_indices = std::vector<size_t>(m_totalBoxes, 0);
        m_colors = GenerateRGBValues(m_totalClasses);
    }
}

void YOLODecoder::EnqueueKernel()
{
    uint32_t startIndex = 0;
    for (size_t featureMapIndex = 0; featureMapIndex < m_inputSize.size(); featureMapIndex++)
    {
        const uint32_t kFeatureSize = (kNumberOfBoxParameters + m_totalClasses + 1) * m_inputSize[featureMapIndex] *
            m_inputSize[featureMapIndex] * m_anchorCount;

        clEnqueueReadBuffer(m_openclWrapper->m_commandQueue,
            m_src[featureMapIndex]->GetBuffer(),
            CL_TRUE,
            0,
            kFeatureSize * sizeof(float),
            m_boxesDeltasScoreLabel.data() + startIndex,
            0,
            0,
            0);
        startIndex += kFeatureSize;
    }
    DeltaToBoxes();
}

inline float YOLODecoder::CalculateIoU(const YOLODecoder::Box &box1, const YOLODecoder::Box &box2)
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

void YOLODecoder::PerformNMS(const std::vector<float> &boxes,
    const std::vector<float> &scores,
    const std::vector<uint8_t> &categories,
    float iouThreshold,
    float scoreThreshold,
    uint16_t maxPossibleBoxes)
{
    m_detectedCount = scores.size();

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

    std::sort(m_indices.begin(),
        m_indices.end(),
        [&](size_t i1, size_t i2) { return m_result[i1].score > m_result[i2].score; });

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

void YOLODecoder::DeltaToBoxes()
{
    size_t boxOffset = 0;
    std::vector<float> kStrides = { 16, 32 };
    for (size_t inputIndex = 0; inputIndex < m_inputSize.size(); inputIndex++)
    {
        for (size_t boxIndex = 0; boxIndex < m_inputSize[inputIndex] * m_inputSize[inputIndex]; boxIndex++)
        {
            for (size_t anchorIndex = 0; anchorIndex < m_anchorCount; anchorIndex++)
            {
                auto sigmoid = [](float x) { return 1.0 / (1.0 + std::exp(-x)); };
                auto myIndex = anchorIndex + (boxIndex + boxOffset) * m_anchorCount;
                auto index = myIndex * (1 + kNumberOfBoxParameters + m_totalClasses);
                const auto dx = m_boxesDeltasScoreLabel[index];
                const auto dy = m_boxesDeltasScoreLabel[index + 1];
                const auto dw = m_boxesDeltasScoreLabel[index + 2];
                const auto dh = m_boxesDeltasScoreLabel[index + 3];
                const auto score = sigmoid(m_boxesDeltasScoreLabel[index + 4]);

                float xGrid = floor(boxIndex / m_inputSize[inputIndex]);
                float yGrid = boxIndex % m_inputSize[inputIndex];
                auto px = (sigmoid(dx) + xGrid) * kStrides[inputIndex];
                auto py = (sigmoid(dy) + yGrid) * kStrides[inputIndex];
                auto pw = std::exp(dw) * kStrides[inputIndex] *
                    m_anchors[inputIndex * kAnchorDimension * m_anchorCount + kAnchorDimension * anchorIndex + 0];
                auto ph = std::exp(dh) * kStrides[inputIndex] *
                    m_anchors[inputIndex * kAnchorDimension * m_anchorCount + kAnchorDimension * anchorIndex + 1];

                auto myBoxIndex = myIndex * kNumberOfBoxParameters;
                auto myCategoryIndex = myIndex * m_totalClasses;
                m_boxes[myBoxIndex] = px - 0.5f * pw;
                m_boxes[myBoxIndex + 1] = py - 0.5f * ph;
                m_boxes[myBoxIndex + 2] = px + 0.5f * pw;
                m_boxes[myBoxIndex + 3] = py + 0.5f * ph;

                uint8_t maxCategoryIndex = 0;
                float maxCategoryProbability = FLT_MIN;
                for (uint8_t categoryIndex = 0; categoryIndex < m_totalClasses; categoryIndex++)
                {
                    const auto category = m_boxesDeltasScoreLabel[index + 5 + categoryIndex];
                    if (category > maxCategoryProbability)
                    {
                        maxCategoryProbability = category;
                        maxCategoryIndex = categoryIndex;
                    }
                }
                m_categories[myIndex] = maxCategoryIndex;
                m_scores[myIndex] = score * sigmoid(maxCategoryProbability);
            }
        }
        boxOffset += m_inputSize[inputIndex] * m_inputSize[inputIndex];
    }
}

void YOLODecoder::SetKernelArguments() {}

void YOLODecoder::DrawBox(uint8_t *outBuffer,
    float x1,
    float y1,
    float x2,
    float y2,
    uint32_t width1,
    uint32_t height1,
    const YOLODecoder::RGB &rgbValue)
{
    auto width = static_cast<int32_t>(width1);
    auto height = static_cast<int32_t>(height1);

    uint32_t xAbsolute1 = static_cast<uint32_t>(std::min(width - 1, std::max(0, static_cast<int32_t>(x1))));
    uint32_t xAbsolute2 = static_cast<uint32_t>(std::min(width - 1, std::max(0, static_cast<int32_t>(x2))));
    uint32_t yAbsolute1 = static_cast<uint32_t>(std::min(height - 1, std::max(0, static_cast<int32_t>(y1))));
    uint32_t yAbsolute2 = static_cast<uint32_t>(std::min(height - 1, std::max(0, static_cast<int32_t>(y2))));

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

void YOLODecoder::GetDimension(std::vector<std::vector<uint32_t>> &dimension)
{
    dimension = { { m_width, m_height, 3 } };
}

void YOLODecoder::CopyOutputBuffer(uint8_t *outBuffer, int32_t)
{
#if 1
    const auto kWhiteColor = YOLODecoder::RGB{ 255, 0, 255 };
    PerformNMS(m_boxes, m_scores, m_categories, 0.1, 0.3);

    for (size_t boxIndex = 0; boxIndex < m_detectedCount; boxIndex++)
    {
        const auto &box = m_filteredBoxes[boxIndex];
        ALOG_GPUML("%d box category %d and probability %f", boxIndex, box.category, box.score);
        auto colorRGB = box.category < m_colors.size() ? m_colors[box.category] : YOLODecoder::RGB{ 255, 255, 255 };
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
            auto colorRGB = labelIndex < m_colors.size() ? m_colors[labelIndex] : YOLODecoder::RGB{ 255, 255, 255 };
            auto v0 = m_boxes[boxIndex * kNumberOfBoxParameters];
            auto v1 = m_boxes[boxIndex * kNumberOfBoxParameters + 1];
            auto v2 = m_boxes[boxIndex * kNumberOfBoxParameters + 2];
            auto v3 = m_boxes[boxIndex * kNumberOfBoxParameters + 3];
            DrawBox(outBuffer, v0, v1, v2, v3, width, height, colorRGB);
        }
    }
#endif
}

void YOLODecoder::CreateBuffers(const std::vector<std::shared_ptr<DataContainerOpenCLFloat>> &src)
{
    m_src = src;
}

void YOLODecoder::FillLayerConstants(const std::filesystem::path &inputPath)
{
    m_anchors.resize(static_cast<size_t>(m_anchorCount) * kAnchorDimension * m_inputSize.size());

    std::vector<int> sWeights;
    std::vector<float> dataWeights;
    auto fullPath = inputPath / (m_name + "_anchors.npy");
    aoba::LoadArrayFromNumpy(fullPath.string(), sWeights, dataWeights);
    std::copy(dataWeights.begin(), dataWeights.end(), m_anchors.data());
}

YOLODecoder::~YOLODecoder() {}
