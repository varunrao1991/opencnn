
#pragma once

#include "layer.h"
#include <string>

class DeconvolutionLayer : public Layer
{
  public:
    explicit DeconvolutionLayer(const std::string &name, std::shared_ptr<OpenclWrapper> openclWrapper);

    void SetKernelArguments();
    void CreateBuffers(const std::vector<std::shared_ptr<DataContainerOpenCLFloat>> &src);
    void FillLayerInputFromFile(const std::filesystem::path &inputPath);
    void SetParameters(const std::vector<std::string> &elements);
    const std::vector<std::string> &GetKernelNames() override;
    void FillLayerConstants(const std::filesystem::path &graphFile) override;

    virtual ~DeconvolutionLayer();

  protected:
    uint32_t m_filterSize[4];
    uint32_t m_inputSize[3];
    uint32_t m_outputSize[3];
    uint32_t m_runCount;

  private:
    DeconvolutionLayer(const DeconvolutionLayer &) = delete;
    DeconvolutionLayer(DeconvolutionLayer &&) = delete;
    DeconvolutionLayer &operator=(const DeconvolutionLayer &) = delete;
    DeconvolutionLayer &operator=(DeconvolutionLayer &&) = delete;
};
