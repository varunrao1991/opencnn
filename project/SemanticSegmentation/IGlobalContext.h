#pragma once

#include <filesystem>

class IGlobalContext
{
  public:
    virtual ~IGlobalContext() = default;
    virtual const std::filesystem::path &GetBasePath() = 0;
    virtual const std::filesystem::path &GetKernelPath() = 0;
    virtual const std::filesystem::path &GetGraphPath() = 0;
    virtual const std::filesystem::path &GetInputsPath() = 0;
    virtual const std::filesystem::path &GetOutputsPath() = 0;
};
