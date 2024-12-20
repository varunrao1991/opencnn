
#pragma once

#include "layer.h"
#include <string>

class DeconvolutionLayer3X3 : public Layer
{
  public:
    explicit DeconvolutionLayer3X3(const std::string &name, std::shared_ptr<OpenclWrapper> openclWrapper);

    void SetKernelArguments();
    void CreateBuffers(const std::vector<std::shared_ptr<DataContainerOpenCLFloat>> &src);
    void FillLayerInputFromFile(const std::filesystem::path &inputPath);
    void SetParameters(const std::vector<std::string> &elements);
    const std::vector<std::string> &GetKernelNames() override;
    void FillLayerConstants(const std::filesystem::path &graphFile) override;
    void EnqueueKernel() override;

    virtual ~DeconvolutionLayer3X3();

  protected:
    uint32_t m_filterSize[4];
    uint32_t m_inputSize[3];
    uint32_t m_runCount;
    uint32_t m_outputSize[3];
    std::unique_ptr<float[]> m_cpuMem;

  private:
    DeconvolutionLayer3X3(const DeconvolutionLayer3X3 &) = delete;
    DeconvolutionLayer3X3(DeconvolutionLayer3X3 &&) = delete;
    DeconvolutionLayer3X3 &operator=(const DeconvolutionLayer3X3 &) = delete;
    DeconvolutionLayer3X3 &operator=(DeconvolutionLayer3X3 &&) = delete;
};
