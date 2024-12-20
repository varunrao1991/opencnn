#pragma once

#include <filesystem>

class IGlobalContext
{
  public:
    virtual ~IGlobalContext() = default;
    virtual const std::filesystem::path &GetKernelPath() = 0;
    virtual const std::filesystem::path &GetModelPath() = 0;
    virtual const std::filesystem::path &GetGraphFile() = 0;
    virtual const std::filesystem::path &GetOutputsPath() = 0;
};
