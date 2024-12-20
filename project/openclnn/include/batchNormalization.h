

#pragma once
#include "dataContainerOpenCLFloat.h"
#include "layer.h"
#include "logger.h"

#include <string>

class BatchNormalization : public Layer
{
  public:
    explicit BatchNormalization(const std::string &name, std::shared_ptr<OpenclWrapper> openclWrapper);

    void SetParameters(const std::vector<std::string> &elements);
    const std::vector<std::string> &GetKernelNames() override;
    void SetKernelArguments();
    void CreateBuffers(const std::vector<std::shared_ptr<DataContainerOpenCLFloat>> &src);
    void FillLayerConstants(const std::filesystem::path &inputPath) override;
    void FillLayerInputFromFile(const std::filesystem::path &inputPath);

    virtual ~BatchNormalization();

  protected:
    uint32_t m_inputSize[3];

  private:
    BatchNormalization(const BatchNormalization &) = delete;
    BatchNormalization(BatchNormalization &&) = delete;
    BatchNormalization &operator=(const BatchNormalization &) = delete;
    BatchNormalization &operator=(BatchNormalization &&) = delete;
};
