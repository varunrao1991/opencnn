#pragma once
#include "IGlobalContext.h"

#include <filesystem>

class GlobalContext : public IGlobalContext
{
  public:
    GlobalContext(const std::filesystem::path &kernelsPath, const std::filesystem::path &modelPath, const std::filesystem::path &outputPath);
    virtual ~GlobalContext();

    const std::filesystem::path &GetKernelPath() override;
    const std::filesystem::path &GetModelPath() override;
    const std::filesystem::path &GetGraphFile() override;
    const std::filesystem::path &GetOutputsPath() override;

  private:
    std::filesystem::path m_kernelsPath;
    std::filesystem::path m_modelPath;
    std::filesystem::path m_graphPath;
    std::filesystem::path m_outputsPath;
};
