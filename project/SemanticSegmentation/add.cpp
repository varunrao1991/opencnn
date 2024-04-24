#include "add.h"
#include <fstream>
#include <string>

Add::Add(const std::string &name, std::shared_ptr<OpenclWrapper> openclWrapper) :
    Layer(name, openclWrapper, 2), m_inputSize{}
{
}

static const std::vector<std::string> s_kKernelNames{ "add" };
const std::vector<std::string> &Add::GetKernelNames()
{
    return s_kKernelNames;
}

void Add::SetParameters(const std::vector<std::string> &elements)
{
    if (elements.size() != 6)
    {
        ALOG_GPUML("Add element size must be 6 represents input dimension and output channels");
        return;
    }
    if (elements[0] == elements[3] && elements[1] == elements[4] && elements[2] == elements[5])
    {
        const auto kTotalSizeElements{ 3 };
        std::vector<int> parameters(kTotalSizeElements);
        int numCount = 0;
        for (int i = 0; i < kTotalSizeElements; i++)
        {
            parameters[numCount++] = std::atoi(elements[i].c_str());
        }
        std::copy(parameters.begin(), parameters.end(), m_inputSize);
    }
    else
    {
        ALOG_GPUML("Add array's size must be same");
    }
}

void Add::SetKernelArguments()
{
    int argCnt = 0;
    m_dimension = 2;
    m_globalSize[0] = m_inputSize[0] * m_inputSize[1];
    m_globalSize[1] = m_inputSize[2] / 4;
    if (m_src.size() != 2)
    {
        ALOG_GPUML("Add : No src memory is created. Failed to set kernel arguments");
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
    clSetKernelArg(m_kernels[0], argCnt++, sizeof(uint32_t), &m_inputSize[0]);
    clSetKernelArg(m_kernels[0], argCnt++, sizeof(uint32_t), &m_inputSize[1]);
    clSetKernelArg(m_kernels[0], argCnt++, sizeof(uint32_t), &m_inputSize[2]);
}

void Add::CreateBuffers(const std::vector<std::shared_ptr<DataContainerOpenCLFloat>> &src)
{
    m_src = src;
    if (m_src.empty())
    {
        auto mem = std::make_shared<DataContainerOpenCLFloat>(
            std::vector{ m_inputSize[0], m_inputSize[1], m_inputSize[2] });
        mem->Allocate(m_openclWrapper->m_context);
        m_src.push_back(mem);

        mem = std::make_shared<DataContainerOpenCLFloat>(
            std::vector{ m_inputSize[0], m_inputSize[1], m_inputSize[2] });
        mem->Allocate(m_openclWrapper->m_context);
        m_src.push_back(mem);
    }
    if (m_dest.empty())
    {
        auto mem = std::make_shared<DataContainerOpenCLFloat>(
            std::vector{ m_inputSize[0], m_inputSize[1], m_inputSize[2] });
        mem->Allocate(m_openclWrapper->m_context);
        m_dest.push_back(mem);
    }
}

Add::~Add() {}
