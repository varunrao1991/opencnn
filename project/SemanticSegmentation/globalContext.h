#pragma once
#include "IGlobalContext.h"

#include <filesystem>

class GlobalContext : public IGlobalContext
{
  public:
    GlobalContext(const std::filesystem::path &basePath);
    virtual ~GlobalContext();

    const std::filesystem::path &GetBasePath() override;
    const std::filesystem::path &GetKernelPath() override;
    const std::filesystem::path &GetGraphPath() override;
    const std::filesystem::path &GetInputsPath() override;
    const std::filesystem::path &GetOutputsPath() override;

  private:
    std::filesystem::path m_basePath;
    std::filesystem::path m_kernelsPath;
    std::filesystem::path m_graphPath;
    std::filesystem::path m_inputsPath;
    std::filesystem::path m_outputsPath;
};
