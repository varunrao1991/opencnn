
#pragma once
#include "layer.h"
#include "logger.h"
#include <string>

class PaddingLayer : public Layer
{
  public:
    explicit PaddingLayer(const std::string &name, std::shared_ptr<OpenclWrapper> openclWrapper);

    void SetParameters(const std::vector<std::string> &elements);
    const std::vector<std::string> &GetKernelNames() override;
    void SetKernelArguments();
    void CreateBuffers(const std::vector<std::shared_ptr<DataContainerOpenCLFloat>> &src);
    void writeOutputBuffers(const std::filesystem::path &outputPath);
    void FillLayerInputFromFile(const std::filesystem::path &inputPath);

    virtual ~PaddingLayer();
    void setPadding();

  protected:
    uint32_t m_inputSize[3];
    uint32_t m_outputSize[3];
    uint32_t m_padding[4];

  private:
    PaddingLayer(const PaddingLayer &) = delete;
    PaddingLayer(PaddingLayer &&) = delete;
    PaddingLayer &operator=(const PaddingLayer &) = delete;
    PaddingLayer &operator=(PaddingLayer &&) = delete;
};
