
#pragma once

#include "layer.h"

class Relu : public Layer
{
  public:
    explicit Relu(const std::string &name, std::shared_ptr<OpenclWrapper> openclWrapper);

    void SetKernelArguments();
    void CreateBuffers(const std::vector<std::shared_ptr<DataContainerOpenCLFloat>> &src);
    void SetParameters(const std::vector<std::string> &elements);
    const std::vector<std::string> &GetKernelNames() override;
    virtual ~Relu();

  protected:
    uint32_t m_inputSize[3];
    uint32_t m_outElements;
    float m_maxValue;

    void SetMaxValue(float);

  private:
    Relu(const Relu &) = delete;
    Relu(Relu &&) = delete;
    Relu &operator=(const Relu &) = delete;
    Relu &operator=(Relu &&) = delete;
};
