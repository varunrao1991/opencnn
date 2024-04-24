#include "globalContext.h"

GlobalContext::GlobalContext(const std::filesystem::path &basePath) :
    IGlobalContext{},
    m_basePath{ basePath },
    m_kernelsPath{ m_basePath / "kernels" },
    m_graphPath{ m_basePath },
    m_inputsPath{ m_basePath / "inputs" },
    m_outputsPath{ m_basePath / "outputs" }
{
}

const std::filesystem::path &GlobalContext::GetBasePath()
{
    return m_basePath;
}

const std::filesystem::path &GlobalContext::GetKernelPath()
{
    return m_kernelsPath;
}

const std::filesystem::path &GlobalContext::GetGraphPath()
{
    return m_graphPath;
}

const std::filesystem::path &GlobalContext::GetInputsPath()
{
    return m_inputsPath;
}

const std::filesystem::path &GlobalContext::GetOutputsPath()
{
    return m_outputsPath;
}

GlobalContext::~GlobalContext() = default;
