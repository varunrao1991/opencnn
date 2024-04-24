
#pragma once

#include "layer.h"
#include <string>

class SigmoidLayer : public Layer
{
  public:
    SigmoidLayer(const std::string &name, std::shared_ptr<OpenclWrapper> openclWrapper);

    void SetKernelArguments();
    void CreateBuffers(const std::vector<std::shared_ptr<DataContainerOpenCLFloat>> &src);
    void SetParameters(const std::vector<std::string> &elements);
    const std::vector<std::string> &GetKernelNames() override;

    virtual ~SigmoidLayer();

  protected:
    uint32_t m_outElements;
    uint32_t m_inputSize[3];

  private:
    SigmoidLayer(const SigmoidLayer &) = delete;
    SigmoidLayer(SigmoidLayer &&) = delete;
    SigmoidLayer &operator=(const SigmoidLayer &) = delete;
    SigmoidLayer &operator=(SigmoidLayer &&) = delete;
};
