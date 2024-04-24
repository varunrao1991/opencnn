
#pragma once

#include "layer.h"

class ImageResize : public Layer
{
  public:
    explicit ImageResize(const std::string &name, std::shared_ptr<OpenclWrapper> openclWrapper);

    void SetKernelArguments();
    void CreateBuffers(const std::vector<std::shared_ptr<DataContainerOpenCLFloat>> &src);
    void SetParameters(const std::vector<std::string> &elements);
    const std::vector<std::string> &GetKernelNames() override;
    virtual ~ImageResize();

  protected:
    std::vector<uint32_t> m_inputSize;
    std::vector<uint32_t> m_outputSize;

  private:
    ImageResize(const ImageResize &) = delete;
    ImageResize(ImageResize &&) = delete;
    ImageResize &operator=(const ImageResize &) = delete;
    ImageResize &operator=(ImageResize &&) = delete;
};
