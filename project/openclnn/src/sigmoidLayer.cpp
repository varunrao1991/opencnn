#include "sigmoidLayer.h"
#include <numeric>

SigmoidLayer::SigmoidLayer(const std::string &name, std::shared_ptr<OpenclWrapper> openclWrapper) :
    Layer(name, openclWrapper), m_inputSize{}, m_outElements{}
{
}

static const std::vector<std::string> s_kKernelNames = { "sigmoid" };
const std::vector<std::string> &SigmoidLayer::GetKernelNames()
{
    return s_kKernelNames;
}

void SigmoidLayer::SetParameters(const std::vector<std::string> &elements)
{
    if (elements.size() != 3)
    {
        ALOG_GPUML("Sigmoid element size must be 3");
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
}

void SigmoidLayer::SetKernelArguments()
{
    int argCnt = 0;
    m_dimension = 1;
    m_globalSize[0] = m_outElements / 16;
    if (m_src.size() != 1)
    {
        ALOG_GPUML("SigmoidLayer: No src memory is created. Failed to set kernel arguments");
        return;
    }

    clSetKernelArg(m_kernels[0], argCnt++, sizeof(cl_mem), &(m_src[0]->GetBuffer()));
    clSetKernelArg(m_kernels[0], argCnt++, sizeof(cl_mem), &(m_dest->GetBuffer()));
    clSetKernelArg(m_kernels[0], argCnt++, sizeof(uint32_t), &m_outElements);
}

void SigmoidLayer::CreateBuffers(const std::vector<std::shared_ptr<DataContainerOpenCLFloat>> &src)
{
    m_src = src;
    if (m_src.empty())
    {
        auto mem = std::make_shared<DataContainerOpenCLFloat>(m_outElements);
        mem->Allocate(m_openclWrapper->m_context);
        m_src.push_back(mem);
    }
    if (m_dest == nullptr)
    {
        auto mem = std::make_shared<DataContainerOpenCLFloat>(m_outElements);
        mem->Allocate(m_openclWrapper->m_context);
        m_dest = mem;
    }
}

SigmoidLayer::~SigmoidLayer() {}
