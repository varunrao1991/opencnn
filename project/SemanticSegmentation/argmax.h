
#pragma once

#include "layer.h"

#include <vector>

class ArgMax : public Layer
{
  public:
    explicit ArgMax(const std::string &name, std::shared_ptr<OpenclWrapper> openclWrapper);
    virtual ~ArgMax();
    const std::vector<std::string> &GetKernelNames() override;
    void SetKernelArguments() override;
    void CreateBuffers(const std::vector<std::shared_ptr<DataContainerOpenCLFloat>> &src);
    void SetParameters(const std::vector<std::string> &elements);

    void CopyOutputBuffer(uint8_t *outBuffer, int32_t index = 0) override;
    void GetDimension(std::vector<std::vector<uint32_t>> &) override;
    void GetOutputTypeSizesInbyte(std::vector<uint32_t> &dimension) override;

  protected:
    uint32_t m_inputSize[3];

  private:
    ArgMax(const ArgMax &) = delete;
    ArgMax(ArgMax &&) = delete;
    ArgMax &operator=(const ArgMax &) = delete;
    ArgMax &operator=(ArgMax &&) = delete;
};
