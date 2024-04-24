
#pragma once
#include "layer.h"
#include <string>

class ConvolutionLayerBias : public Layer
{
  public:
    explicit ConvolutionLayerBias(const std::string &name, std::shared_ptr<OpenclWrapper> openclWrapper);

    void SetParameters(const std::vector<std::string> &elements);
    void EnableReluActivation();
    void EnqueueKernel() override;
    const std::vector<std::string> &GetKernelNames() override;
    void SetKernelArguments();
    void CreateBuffers(const std::vector<std::shared_ptr<DataContainerOpenCLFloat>> &src);
    void FillLayerConstants(const std::filesystem::path &inputPath) override;
    void FillLayerInputFromFile(const std::filesystem::path &inputPath);

    virtual ~ConvolutionLayerBias();

  protected:
    uint32_t m_filterSize[4];
    uint32_t m_inputSize[3];
    uint32_t m_outputSize[3];
    uint32_t m_stride;
    char m_enableRelu;
    bool m_enableTranspose;

  private:
    ConvolutionLayerBias(const ConvolutionLayerBias &) = delete;
    ConvolutionLayerBias(ConvolutionLayerBias &&) = delete;
    ConvolutionLayerBias &operator=(const ConvolutionLayerBias &) = delete;
    ConvolutionLayerBias &operator=(ConvolutionLayerBias &&) = delete;
};
