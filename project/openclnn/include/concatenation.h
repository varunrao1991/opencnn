
#pragma once

#include "layer.h"

#include <cstdint>

class Concatenation : public Layer
{
  public:
    explicit Concatenation(const std::string &name,
        std::shared_ptr<OpenclWrapper> openclWrapper);
    void SetParameters(const std::vector<std::string> &elements);
    const std::vector<std::string> &GetKernelNames() override;
    void SetKernelArguments();
    void CreateBuffers(const std::vector<std::shared_ptr<DataContainerOpenCLFloat>> &src);

    virtual ~Concatenation();

  protected:
    uint32_t m_inputSize1[3];
    uint32_t m_inputSize2[3];

  private:
    Concatenation(const Concatenation &) = delete;
    Concatenation(Concatenation &&) = delete;
    Concatenation &operator=(const Concatenation &) = delete;
    Concatenation &operator=(Concatenation &&) = delete;
};
