
#pragma once
#include "layer.h"
#include <string>

class PointwiseConvolutionLayer : public Layer
{
  public:
    explicit PointwiseConvolutionLayer(const std::string &name, std::shared_ptr<OpenclWrapper> openclWrapper);

    const std::vector<std::string> &GetKernelNames() override;

    void SetParameters(const std::vector<std::string> &elements);
    void EnableReluActivation();
    void SetKernelArguments();
    void CreateBuffers(const std::vector<std::shared_ptr<DataContainerOpenCLFloat>> &src);
    void FillLayerConstants(const std::filesystem::path &graphFile) override;
    void FillLayerInputFromFile(const std::filesystem::path &inputPath);

    virtual ~PointwiseConvolutionLayer();

  protected:
    uint32_t m_filterSize[2];
    uint32_t m_inputSize[3];
    char m_enableRelu;

  private:
    PointwiseConvolutionLayer(const PointwiseConvolutionLayer &) = delete;
    PointwiseConvolutionLayer(PointwiseConvolutionLayer &&) = delete;
    PointwiseConvolutionLayer &operator=(const PointwiseConvolutionLayer &) = delete;
    PointwiseConvolutionLayer &operator=(PointwiseConvolutionLayer &&) = delete;
};
