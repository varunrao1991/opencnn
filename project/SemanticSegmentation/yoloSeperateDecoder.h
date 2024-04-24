
#pragma once

#include "layer.h"

#include <memory>

class YOLOSeperateDecoder : public Layer
{
  public:
    struct Box
    {
        float x1, y1, x2, y2;
        float score;
        uint8_t category;
    };
    struct RGB
    {
        uint8_t red;
        uint8_t green;
        uint8_t blue;
    };

    std::vector<RGB> GenerateRGBValues(uint32_t totalClasses);
    explicit YOLOSeperateDecoder(const std::string &name, std::shared_ptr<OpenclWrapper> openclWrapper);
    void SetParameters(const std::vector<std::string> &elements);
    const std::vector<std::string> &GetKernelNames() override;
    void CreateBuffers(const std::vector<std::shared_ptr<DataContainerOpenCLFloat>> &src);
    void CopyOutputBuffer(uint8_t *outBuffer, int32_t);
    void GetDimension(std::vector<std::vector<uint32_t>> &) override;
    void GetOutputTypeSizesInbyte(std::vector<uint32_t> &) override;

    virtual ~YOLOSeperateDecoder();

  protected:
    uint32_t m_width;
    uint32_t m_height;
    uint32_t m_totalClasses;
    uint32_t m_anchorCount;
    uint32_t m_totalBoxes;
    std::vector<uint32_t> m_inputSize;
    std::vector<float> m_anchors;
    std::vector<float> m_boxes;
    std::vector<float> m_boxesDeltasScore;
    std::vector<float> m_labels;
    std::vector<uint8_t> m_categories;
    std::vector<float> m_scores;
    std::vector<RGB> m_colors;

    std::vector<size_t> m_indices;
    std::vector<bool> m_keep;
    std::vector<Box> m_filteredBoxes;
    std::vector<Box> m_result;
    uint32_t m_detectedCount;

    void DeltaToBoxes();
    void SetKernelArguments() override;
    void EnqueueKernel() override;
    float CalculateIoU(const Box &box1, const Box &box2);
    void FillLayerConstants(const std::filesystem::path &graphFile) override;
    void PerformNMS(const std::vector<float> &boxes,
        const std::vector<float> &scores,
        const std::vector<uint8_t> &categories,
        float iouThreshold,
        float scoreThreshold,
        uint16_t maxPossibleBoxes = 200);

    void DrawBox(uint8_t *outBuffer,
        float x1,
        float x2,
        float y1,
        float y2,
        uint32_t width,
        uint32_t height,
        const RGB &rgbValue);

  private:
    YOLOSeperateDecoder(const YOLOSeperateDecoder &) = delete;
    YOLOSeperateDecoder(YOLOSeperateDecoder &&) = delete;
    YOLOSeperateDecoder &operator=(const YOLOSeperateDecoder &) = delete;
    YOLOSeperateDecoder &operator=(YOLOSeperateDecoder &&) = delete;
};
