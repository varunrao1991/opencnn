
#pragma once

#include "layer.h"

class LeakyRelu : public Layer
{
  public:
    explicit LeakyRelu(const std::string &name, std::shared_ptr<OpenclWrapper> openclWrapper);

    void SetKernelArguments();
    void CreateBuffers(const std::vector<std::shared_ptr<DataContainerOpenCLFloat>> &src);
    void SetParameters(const std::vector<std::string> &elements);
    const std::vector<std::string> &GetKernelNames() override;
    virtual ~LeakyRelu();

  protected:
    uint32_t m_inputSize[3];
    float m_leakValue;

    void SetLeakValue(float);

  private:
    LeakyRelu(const LeakyRelu &) = delete;
    LeakyRelu(LeakyRelu &&) = delete;
    LeakyRelu &operator=(const LeakyRelu &) = delete;
    LeakyRelu &operator=(LeakyRelu &&) = delete;
};
