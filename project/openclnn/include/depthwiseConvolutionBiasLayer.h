
#pragma once
#include "layer.h"
#include "logger.h"
#include <string>

class DepthwiseConvolutionBiasLayer : public Layer
{
  public:
    explicit DepthwiseConvolutionBiasLayer(const std::string &name, std::shared_ptr<OpenclWrapper> openclWrapper);

    void SetParameters(const std::vector<std::string> &elements);
    const std::vector<std::string> &GetKernelNames() override;
    void SetKernelArguments();
    void CreateBuffers(const std::vector<std::shared_ptr<DataContainerOpenCLFloat>> &src);
    void FillLayerConstants(const std::filesystem::path &inputPath) override;
    void writeOutputBuffers(const std::filesystem::path &basePath);
    void FillLayerInputFromFile(const std::filesystem::path &inputPath);

    virtual ~DepthwiseConvolutionBiasLayer();

  protected:
    uint32_t m_filterSize[3];
    uint32_t m_inputSize[3];
    uint32_t m_outputSize[3];
    uint32_t m_stride;

  private:
    DepthwiseConvolutionBiasLayer(const DepthwiseConvolutionBiasLayer &) = delete;
    DepthwiseConvolutionBiasLayer(DepthwiseConvolutionBiasLayer &&) = delete;
    DepthwiseConvolutionBiasLayer &operator=(const DepthwiseConvolutionBiasLayer &) = delete;
    DepthwiseConvolutionBiasLayer &operator=(DepthwiseConvolutionBiasLayer &&) = delete;
};
