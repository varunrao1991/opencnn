
#pragma once

#include "layer.h"

#include <memory>

struct Box
{
    float x1, y1, x2, y2;
    float score;
    uint8_t category;
};

class SSDDecoder : public Layer
{
  public:
    struct RGB
    {
        uint8_t red;
        uint8_t green;
        uint8_t blue;
    };

    std::vector<RGB> GenerateRGBValues(uint32_t m_totalClasses);
    explicit SSDDecoder(const std::string &name, std::shared_ptr<OpenclWrapper> openclWrapper);
    void SetParameters(const std::vector<std::string> &elements);
    const std::vector<std::string> &GetKernelNames() override;
    void DeltaToBoxes();
    void SetKernelArguments() override;
    void CreateBuffers(const std::vector<std::shared_ptr<DataContainerOpenCLFloat>> &src);
    void EnqueueKernel() override;

    void ignoreBackground(bool enable = true);
    void FillLayerConstants(const std::filesystem::path &inputPath);
    void PerformNMS(const std::vector<float> &boxes,
        const std::vector<float> &scores,
        const std::vector<uint8_t> &categories,
        float iouThreshold,
        float scoreThreshold,
        uint16_t maxPossibleBoxes = 200);

    void DisplayOutputBuffer() override;
    void CopyOutputBuffer(uint8_t *outBuffer, int32_t);
    void DrawBox(uint8_t *outBuffer,
        float x1,
        float x2,
        float y1,
        float y2,
        uint32_t width,
        uint32_t height,
        const SSDDecoder::RGB &rgbValue);

    void GetDimension(std::vector<std::vector<uint32_t>> &) override;

    virtual ~SSDDecoder();

  protected:
    uint32_t m_width;
    uint32_t m_height;
    uint32_t m_totalClasses;
    uint32_t m_totalBoxes;
    std::vector<float> m_priorBoxes;
    std::vector<double> m_variances;
    std::vector<uint32_t> m_inputSize;
    std::vector<float> m_boxes;
    std::vector<float> m_boxesDeltas;
    std::vector<float> m_labels;
    std::vector<uint8_t> m_categories;
    std::vector<float> m_scores;
    std::vector<SSDDecoder::RGB> m_colors;

    std::vector<size_t> m_indices;
    std::vector<bool> m_keep;
    std::vector<Box> m_filteredBoxes;
    std::vector<Box> m_result;
    uint32_t m_detectedCount;

  private:
    SSDDecoder(const SSDDecoder &) = delete;
    SSDDecoder(SSDDecoder &&) = delete;
    SSDDecoder &operator=(const SSDDecoder &) = delete;
    SSDDecoder &operator=(SSDDecoder &&) = delete;
};
