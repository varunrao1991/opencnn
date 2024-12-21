#include "softmax.h"
#include <fstream>
#include <string>

#include <iomanip>
#include <numeric>

Softmax::Softmax(const std::string &name, std::shared_ptr<OpenclWrapper> openclWrapper) :
    Layer(name, openclWrapper, 2), m_inputSize{}, m_intermediateMemory{ nullptr }
{
}

static const std::vector<std::string> s_kKernelNames = { "softmax1", "softmax2" };
const std::vector<std::string> &Softmax::GetKernelNames()
{
    return s_kKernelNames;
}

void Softmax::SetParameters(const std::vector<std::string> &elements)
{
    if (elements.size() != 2 && elements.size() != 3)
    {
        ALOG_GPUML("Softmax element size must be 3 represents input dimension and output channels");
        return;
    }
    else
    {
        auto kTotalSizeElements = elements.size();
        std::vector<int> parameters(kTotalSizeElements);
        int numCount = 0;
        for (int i = 0; i < kTotalSizeElements; i++)
        {
            parameters[numCount++] = std::atoi(elements[i].c_str());
        }
        if (elements.size() == 3)
        {
            m_inputSize[0] = parameters[0];
            m_inputSize[1] = parameters[1];
            m_inputSize[2] = parameters[2];
        }
        else
        {
            m_inputSize[0] = parameters[0];
            m_inputSize[1] = 1;
            m_inputSize[2] = parameters[1];
        }
    }
}

void Softmax::EnqueueKernel()
{
    SetKernelArguments();
    m_dimension = 1;
    m_globalSize[0] = m_inputSize[0] * m_inputSize[1];

    cl_int err = clEnqueueNDRangeKernel(
        m_openclWrapper->m_commandQueue, m_kernels[0], m_dimension, m_globalOffset, m_globalSize, 0, 0, 0, nullptr);
    ALOG_GPUML_CHECK_ERROR1(err, __LINE__, "clEnqueueNDRangeKernel");

    m_dimension = 2;
    m_globalSize[0] = m_inputSize[0] * m_inputSize[1];
    m_globalSize[1] = m_inputSize[2];

    err = clEnqueueNDRangeKernel(
        m_openclWrapper->m_commandQueue, m_kernels[1], m_dimension, m_globalOffset, m_globalSize, 0, 0, 0, nullptr);
    ALOG_GPUML_CHECK_ERROR1(err, __LINE__, "clEnqueueNDRangeKernel");
}

void Softmax::SetKernelArguments()
{
    if (m_src.size() != 1)
    {
        ALOG_GPUML("Softmax : No src memory is created. Failed to set kernel arguments");
        return;
    }

    int argCnt = 0;
    clSetKernelArg(m_kernels[0], argCnt++, sizeof(cl_mem), &(m_src[0]->GetBuffer()));
    clSetKernelArg(m_kernels[0], argCnt++, sizeof(cl_mem), &(m_intermediateMemory->GetBuffer()));
    clSetKernelArg(m_kernels[0], argCnt++, sizeof(uint32_t), &m_inputSize[2]);

    argCnt = 0;
    clSetKernelArg(m_kernels[1], argCnt++, sizeof(cl_mem), &(m_src[0]->GetBuffer()));
    clSetKernelArg(m_kernels[1], argCnt++, sizeof(cl_mem), &(m_intermediateMemory->GetBuffer()));
    clSetKernelArg(m_kernels[1], argCnt++, sizeof(cl_mem), &(m_dest->GetBuffer()));
    clSetKernelArg(m_kernels[1], argCnt++, sizeof(uint32_t), &m_inputSize[2]);
}

void Softmax::CreateBuffers(const std::vector<std::shared_ptr<DataContainerOpenCLFloat>> &src)
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

    m_intermediateMemory = std::make_shared<DataContainerOpenCLFloat>(std::vector{ m_inputSize[0], m_inputSize[1] });
    m_intermediateMemory->Allocate(m_openclWrapper->m_context);
}

Softmax::~Softmax() {}
