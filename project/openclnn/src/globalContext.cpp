#include "globalContext.h"

GlobalContext::GlobalContext(const std::filesystem::path &kernelsPath, const std::filesystem::path &modelPath, const std::filesystem::path &outputPath) :
    IGlobalContext{},
    m_kernelsPath{ kernelsPath },
    m_modelPath{ modelPath },
    m_graphPath{ modelPath / (modelPath.filename().string() + ".txt") },
    m_outputsPath{ outputPath }
{
}

const std::filesystem::path &GlobalContext::GetKernelPath()
{
    return m_kernelsPath;
}

const std::filesystem::path &GlobalContext::GetModelPath()
{
    return m_modelPath;
}

const std::filesystem::path &GlobalContext::GetGraphFile()
{
    return m_graphPath;
}

const std::filesystem::path &GlobalContext::GetOutputsPath()
{
    return m_outputsPath;
}

GlobalContext::~GlobalContext() = default;
