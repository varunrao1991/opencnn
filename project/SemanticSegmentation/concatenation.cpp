#include "concatenation.h"
#include <fstream>
#include <string>

#include <iomanip>
#include <numeric>

Concatenation::Concatenation(const std::string &name, std::shared_ptr<OpenclWrapper> openclWrapper) :
    Layer(name, openclWrapper, 2), m_inputSize1{}, m_inputSize2{}
{
}

static const std::vector<std::string> s_kKernelNames = { "concatenation" };
const std::vector<std::string> &Concatenation::GetKernelNames()
{
    return s_kKernelNames;
}

void Concatenation::SetParameters(const std::vector<std::string> &elements)
{
    if (elements.size() != 6)
    {
        ALOG_GPUML("Concatenation element size must be 6 represents input dimension and output channels");
        return;
    }
    if (elements[0] == elements[3] && elements[1] == elements[4])
    {
        auto kTotalSizeElements = 6;
        std::vector<int> parameters(kTotalSizeElements);
        int numCount = 0;
        for (int i = 0; i < kTotalSizeElements; i++)
        {
            parameters[numCount++] = std::atoi(elements[i].c_str());
        }

        std::copy(parameters.begin(), parameters.begin() + 3, m_inputSize1);
        std::copy(parameters.begin() + 3, parameters.end(), m_inputSize2);
    }
    else
    {
        ALOG_GPUML("Concatenation array's size must be same");
    }
}
void Concatenation::SetKernelArguments()
{
    int argCnt = 0;
    m_dimension = 2;
    m_globalSize[0] = m_inputSize1[0] * m_inputSize1[1];
    m_globalSize[1] = m_inputSize1[2] + m_inputSize2[2];
    if (m_src.size() != 2)
    {
        ALOG_GPUML("Concatenation : No src memory is created. Failed to set kernel arguments");
        return;
    }
    if (m_dest.size() != 1)
    {
        ALOG_GPUML("No dest memory is created. Failed to set kernel arguments");
        return;
    }
    clSetKernelArg(m_kernels[0], argCnt++, sizeof(cl_mem), &(m_src[0]->GetBuffer()));
    clSetKernelArg(m_kernels[0], argCnt++, sizeof(cl_mem), &(m_src[1]->GetBuffer()));
    clSetKernelArg(m_kernels[0], argCnt++, sizeof(cl_mem), &(m_dest[0]->GetBuffer()));
    clSetKernelArg(m_kernels[0], argCnt++, sizeof(uint32_t), &m_inputSize1[0]);
    clSetKernelArg(m_kernels[0], argCnt++, sizeof(uint32_t), &m_inputSize1[1]);
    clSetKernelArg(m_kernels[0], argCnt++, sizeof(uint32_t), &m_inputSize1[2]);
    clSetKernelArg(m_kernels[0], argCnt++, sizeof(uint32_t), &m_inputSize2[2]);
}
void Concatenation::CreateBuffers(const std::vector<std::shared_ptr<DataContainerOpenCLFloat>> &src)
{
    m_src = src;
    uint32_t totalChannels = m_inputSize1[2] + m_inputSize2[2];
    if (m_src.empty())
    {
        auto mem = std::make_shared<DataContainerOpenCLFloat>(
            std::vector{ m_inputSize1[0], m_inputSize1[1], m_inputSize1[2] });
        mem->Allocate(m_openclWrapper->m_context);
        m_src.push_back(mem);
        mem = std::make_shared<DataContainerOpenCLFloat>(
            std::vector{ m_inputSize2[0], m_inputSize2[1], m_inputSize2[2] });
        mem->Allocate(m_openclWrapper->m_context);
        m_src.push_back(mem);
    }
    if (m_dest.empty())
    {
        auto mem = std::make_shared<DataContainerOpenCLFloat>(
            std::vector{ m_inputSize2[0], m_inputSize2[1], totalChannels });
        mem->Allocate(m_openclWrapper->m_context);
        m_dest.push_back(mem);
    }
}

Concatenation::~Concatenation() {}
