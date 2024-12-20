
#pragma once
#include "layer.h"
#include "logger.h"
#include <string>

class BatchNormalization1 : public Layer
{
  public:
    explicit BatchNormalization1(const std::string &name, std::shared_ptr<OpenclWrapper> openclWrapper);

    void SetParameters(const std::vector<std::string> &elements);
    const std::vector<std::string> &GetKernelNames() override;
    void SetKernelArguments();
    void CreateBuffers(const std::vector<std::shared_ptr<DataContainerOpenCLFloat>> &src);
    void FillLayerConstants(const std::filesystem::path &graphFile) override;
    void FillLayerInputFromFile(const std::filesystem::path &inputPath);

    virtual ~BatchNormalization1();

  protected:
    uint32_t m_inputSize[3];

  private:
    BatchNormalization1(const BatchNormalization1 &) = delete;
    BatchNormalization1(BatchNormalization1 &&) = delete;
    BatchNormalization1 &operator=(const BatchNormalization1 &) = delete;
    BatchNormalization1 &operator=(BatchNormalization1 &&) = delete;
};
