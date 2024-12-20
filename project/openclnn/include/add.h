
#pragma once

#include "layer.h"

class Add : public Layer
{
  public:
    explicit Add(const std::string &name, std::shared_ptr<OpenclWrapper> openclWrapper);
    void SetParameters(const std::vector<std::string> &elements);
    void SetKernelArguments();
    void CreateBuffers(const std::vector<std::shared_ptr<DataContainerOpenCLFloat>> &src);

    const std::vector<std::string> &GetKernelNames() override;
    virtual ~Add();

  protected:
    uint32_t m_inputSize[3];

  private:
    Add(const Add &) = delete;
    Add(Add &&) = delete;
    Add &operator=(const Add &) = delete;
    Add &operator=(Add &&) = delete;
};