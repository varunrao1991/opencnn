
#pragma once
#include "layer.h"
#include <string>

class MaxPool : public Layer
{
  public:
    explicit MaxPool(const std::string &name, std::shared_ptr<OpenclWrapper> openclWrapper);

    void SetParameters(const std::vector<std::string> &elements);
    const std::vector<std::string> &GetKernelNames() override;
    void SetKernelArguments();
    void CreateBuffers(const std::vector<std::shared_ptr<DataContainerOpenCLFloat>> &src);
    void FillLayerInputFromFile(const std::filesystem::path &inputPath);

    virtual ~MaxPool();

  protected:
    uint32_t m_inputSize[3];
    uint32_t m_stride;
    uint32_t m_outputSize[3];

  private:
    MaxPool(const MaxPool &) = delete;
    MaxPool(MaxPool &&) = delete;
    MaxPool &operator=(const MaxPool &) = delete;
    MaxPool &operator=(MaxPool &&) = delete;
};
