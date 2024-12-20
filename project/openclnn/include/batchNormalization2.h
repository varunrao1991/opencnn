
#pragma once
#include "layer.h"
#include <string>

class BatchNormalization2 : public Layer
{
  public:
    explicit BatchNormalization2(const std::string &name, std::shared_ptr<OpenclWrapper> openclWrapper);

    void SetParameters(const std::vector<std::string> &elements);
    const std::vector<std::string> &GetKernelNames() override;
    void SetKernelArguments();
    void CreateBuffers(const std::vector<std::shared_ptr<DataContainerOpenCLFloat>> &src);
    void FillLayerConstants(const std::filesystem::path &graphFile) override;
    void FillLayerInputFromFile(const std::filesystem::path &inputPath);

    virtual ~BatchNormalization2();

  protected:
    uint32_t m_inputSize[3];

  private:
    BatchNormalization2(const BatchNormalization2 &) = delete;
    BatchNormalization2(BatchNormalization2 &&) = delete;
    BatchNormalization2 &operator=(const BatchNormalization2 &) = delete;
    BatchNormalization2 &operator=(BatchNormalization2 &&) = delete;
};
