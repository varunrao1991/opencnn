
#pragma once
#include "layer.h"
#include <string>

class DepthwiseConvolutionLayer : public Layer
{
  public:
    explicit DepthwiseConvolutionLayer(const std::string &name, std::shared_ptr<OpenclWrapper> openclWrapper);

    void SetParameters(const std::vector<std::string> &elements);
    const std::vector<std::string> &GetKernelNames() override;
    void SetKernelArguments();
    void CreateBuffers(const std::vector<std::shared_ptr<DataContainerOpenCLFloat>> &src);
    void FillLayerConstants(const std::filesystem::path &graphFile) override;
    void writeInputBuffers(const std::filesystem::path &outputPath);
    void writeOutputBuffers(const std::filesystem::path &outputPath);
    void FillLayerInputFromFile(const std::filesystem::path &inputPath);

    virtual ~DepthwiseConvolutionLayer();

  protected:
    uint32_t m_filterSize[3];
    uint32_t m_inputSize[3];
    uint32_t m_outputSize[3];
    uint32_t m_stride;

  private:
    DepthwiseConvolutionLayer(const DepthwiseConvolutionLayer &) = delete;
    DepthwiseConvolutionLayer(DepthwiseConvolutionLayer &&) = delete;
    DepthwiseConvolutionLayer &operator=(const DepthwiseConvolutionLayer &) = delete;
    DepthwiseConvolutionLayer &operator=(DepthwiseConvolutionLayer &&) = delete;
};
