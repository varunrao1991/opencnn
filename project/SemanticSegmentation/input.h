
#pragma once

#include "layer.h"

#include <array>
#include <string>
#include <windows.h>

class InputLayer : public Layer
{
  public:
    explicit InputLayer(const std::string &name, std::shared_ptr<OpenclWrapper> openclWrapper);

    void SetKernelArguments() override;
    void CreateBuffers(const std::vector<std::shared_ptr<DataContainerOpenCLFloat>> &src) override;
    void SetParameters(const std::vector<std::string> &elements) override;
    void GetDimension(uint32_t &width, uint32_t &height);
    const std::vector<std::string> &GetKernelNames() override;
    void FillInputFromBuffer(float *dataIn);
    void FillLayerInputFromFile(const std::filesystem::path &inputPath);
    void GetDimension(std::vector<std::vector<uint32_t>> &) override;

    virtual ~InputLayer();

  private:
    uint32_t m_inputSize[3];

  private:
    InputLayer(const InputLayer &) = delete;
    InputLayer(InputLayer &&) = delete;
    InputLayer &operator=(const InputLayer &) = delete;
    InputLayer &operator=(InputLayer &&) = delete;
};
