
#pragma once
#include "layer.h"
#include <string>

class PointwiseConvolutionLayerBias : public Layer
{
  public:
    explicit PointwiseConvolutionLayerBias(const std::string &name, std::shared_ptr<OpenclWrapper> openclWrapper);

    void SetParameters(const std::vector<std::string> &elements);
    void EnqueueKernel() override;
    const std::vector<std::string> &GetKernelNames() override;
    void EnableReluActivation();
    void SetKernelArguments();
    void CreateBuffers(const std::vector<std::shared_ptr<DataContainerOpenCLFloat>> &src);
    void FillLayerConstants(const std::filesystem::path &graphFile) override;
    void FillLayerInputFromFile(const std::filesystem::path &inputPath);

    virtual ~PointwiseConvolutionLayerBias();

  protected:
    const size_t *getLocalWorkSize() { return nullptr; }

    uint32_t m_filterSize[2];
    uint32_t m_inputSize[3];
    uint32_t m_outputSize[3];
    char m_enableRelu;
    uint32_t m_useEven;

  private:
    PointwiseConvolutionLayerBias(const PointwiseConvolutionLayerBias &) = delete;
    PointwiseConvolutionLayerBias(PointwiseConvolutionLayerBias &&) = delete;
    PointwiseConvolutionLayerBias &operator=(const PointwiseConvolutionLayerBias &) = delete;
    PointwiseConvolutionLayerBias &operator=(PointwiseConvolutionLayerBias &&) = delete;
};
