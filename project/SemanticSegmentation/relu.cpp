#include "relu.h"
#include <iomanip>
#include <numeric>

Relu::Relu(const std::string &name, std::shared_ptr<OpenclWrapper> openclWrapper) :
    Layer(name, openclWrapper), m_inputSize{}, m_maxValue{}, m_outElements{}
{
}

static const std::vector<std::string> s_kKernelNames = { "relu" };
const std::vector<std::string> &Relu::GetKernelNames()
{
    return s_kKernelNames;
}

void Relu::SetParameters(const std::vector<std::string> &elements)
{
    if (elements.size() != 4 && elements.size() != 5)
    {
        ALOG_GPUML("Relu element size must be 4");
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
    m_outElements = std::accumulate(parameters.begin(), parameters.end(), 1, std::multiplies<int>());

    SetMaxValue(std::atof(elements[kTotalSizeElements].c_str()));
}

void Relu::SetMaxValue(float value)
{
    m_maxValue = value;
}

void Relu::SetKernelArguments()
{
    int argCnt = 0;
    m_dimension = 1;
    m_globalSize[0] = m_outElements / 16;
    if (m_src.size() != 1)
    {
        ALOG_GPUML("Relu : No src memory is created. Failed to set kernel arguments");
        return;
    }
    if (m_dest.size() != 1)
    {
        ALOG_GPUML("No dest memory is created. Failed to set kernel arguments");
        return;
    }
    clSetKernelArg(m_kernels[0], argCnt++, sizeof(cl_mem), &(m_src[0]->GetBuffer()));
    clSetKernelArg(m_kernels[0], argCnt++, sizeof(cl_mem), &(m_dest[0]->GetBuffer()));
    clSetKernelArg(m_kernels[0], argCnt++, sizeof(uint32_t), &m_outElements);
    clSetKernelArg(m_kernels[0], argCnt++, sizeof(cl_float), &m_maxValue);
}

void Relu::CreateBuffers(const std::vector<std::shared_ptr<DataContainerOpenCLFloat>> &src)
{
    m_src = src;
    if (m_src.empty())
    {
        auto mem = std::make_shared<DataContainerOpenCLFloat>(m_outElements);
        mem->Allocate(m_openclWrapper->m_context);
        m_src.push_back(mem);
    }
    if (m_dest.empty())
    {
        auto mem = std::make_shared<DataContainerOpenCLFloat>(m_outElements);
        mem->Allocate(m_openclWrapper->m_context);
        m_dest.push_back(mem);
    }
}

Relu::~Relu() {}
