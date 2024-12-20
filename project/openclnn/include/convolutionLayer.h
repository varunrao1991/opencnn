
#pragma once
#include "layer.h"
#include <string>

class ConvolutionLayer : public Layer
{
  public:
    explicit ConvolutionLayer(const std::string &name, std::shared_ptr<OpenclWrapper> openclWrapper);
    void SetParameters(const std::vector<std::string> &elements);
    const std::vector<std::string> &GetKernelNames() override;
    void SetKernelArguments();
    void CreateBuffers(const std::vector<std::shared_ptr<DataContainerOpenCLFloat>> &src);
    void FillLayerConstants(const std::filesystem::path &graphFile) override;
    void FillLayerInputFromFile(const std::filesystem::path &inputPath);

    virtual ~ConvolutionLayer();

  protected:
    void EnableReluActivation();

    uint32_t m_filterSize[4];
    uint32_t m_inputSize[3];
    uint32_t m_outputSize[3];
    uint32_t m_stride;
    char m_enableRelu;

  private:
    ConvolutionLayer(const ConvolutionLayer &) = delete;
    ConvolutionLayer(ConvolutionLayer &&) = delete;
    ConvolutionLayer &operator=(const ConvolutionLayer &) = delete;
    ConvolutionLayer &operator=(ConvolutionLayer &&) = delete;
};
