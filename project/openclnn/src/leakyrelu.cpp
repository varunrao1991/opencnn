#include "leakyrelu.h"

#include <iomanip>
#include <numeric>

LeakyRelu::LeakyRelu(const std::string &name, std::shared_ptr<OpenclWrapper> openclWrapper) :
    Layer(name, openclWrapper), m_inputSize{}, m_leakValue{ 0.0f }
{
}

static const std::vector<std::string> s_kKernelNames = { "leakyrelu" };
const std::vector<std::string> &LeakyRelu::GetKernelNames()
{
    return s_kKernelNames;
}

void LeakyRelu::SetParameters(const std::vector<std::string> &elements)
{
    if (elements.size() != 4 && elements.size() != 5)
    {
        ALOG_GPUML("LeakyRelu element size must be 4");
        return;
    }

    auto kTotalSizeElements = 3;
    std::vector<int> parameters(kTotalSizeElements);
    int numCount = 0;
    for (int i = 0; i < kTotalSizeElements; i++)
    {
        parameters[numCount++] = std::atoi(elements[i].c_str());
    }
    std::copy(parameters.begin(), parameters.end(), m_inputSize);

    SetLeakValue(std::atof(elements[kTotalSizeElements].c_str()));
}

void LeakyRelu::SetLeakValue(float value)
{
    m_leakValue = value;
}

void LeakyRelu::SetKernelArguments()
{
    int argCnt = 0;
    m_dimension = 2;
    m_globalSize[0] = m_inputSize[0] * m_inputSize[1];
    m_globalSize[1] = (m_inputSize[2] + 15) / 16;
    if (m_src.size() != 1)
    {
        ALOG_GPUML("LeakyRelu : No src memory is created. Failed to set kernel arguments");
        return;
    }

    clSetKernelArg(m_kernels[0], argCnt++, sizeof(cl_mem), &(m_src[0]->GetBuffer()));
    clSetKernelArg(m_kernels[0], argCnt++, sizeof(cl_mem), &(m_dest->GetBuffer()));
    clSetKernelArg(m_kernels[0], argCnt++, sizeof(uint32_t), &m_inputSize[0]);
    clSetKernelArg(m_kernels[0], argCnt++, sizeof(uint32_t), &m_inputSize[1]);
    clSetKernelArg(m_kernels[0], argCnt++, sizeof(uint32_t), &m_inputSize[2]);
    clSetKernelArg(m_kernels[0], argCnt++, sizeof(cl_float), &m_leakValue);
}

void LeakyRelu::CreateBuffers(const std::vector<std::shared_ptr<DataContainerOpenCLFloat>> &src)
{
    m_src = src;
    if (m_src.empty())
    {
        auto mem =
            std::make_shared<DataContainerOpenCLFloat>(std::vector{ m_inputSize[0], m_inputSize[1], m_inputSize[2] });
        mem->Allocate(m_openclWrapper->m_context);
        m_src.push_back(mem);
    }
    if (m_dest == nullptr)
    {
        auto mem =
            std::make_shared<DataContainerOpenCLFloat>(std::vector{ m_inputSize[0], m_inputSize[1], m_inputSize[2] });
        mem->Allocate(m_openclWrapper->m_context);
        m_dest = mem;
    }
}

LeakyRelu::~LeakyRelu() {}
