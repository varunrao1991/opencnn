
#pragma once

#include "layer.h"

class Softmax : public Layer
{
  public:
    explicit Softmax(const std::string &name, std::shared_ptr<OpenclWrapper> openclWrapper);
    void SetParameters(const std::vector<std::string> &elements);
    const std::vector<std::string> &GetKernelNames() override;
    void SetKernelArguments() override;
    void CreateBuffers(const std::vector<std::shared_ptr<DataContainerOpenCLFloat>> &src);
    void EnqueueKernel() override;

    virtual ~Softmax();

  protected:
    uint32_t m_inputSize[3];
    std::shared_ptr<DataContainerOpenCLFloat> m_intermediateMemory;

  private:
    Softmax(const Softmax &) = delete;
    Softmax(Softmax &&) = delete;
    Softmax &operator=(const Softmax &) = delete;
    Softmax &operator=(Softmax &&) = delete;
};
