#include "argmax.h"
#include "bmpreader.h"

#include <array>
#include <fstream>

void normalize(float *buffer, size_t size)
{
    float min_val = std::numeric_limits<float>::max();
    float max_val = std::numeric_limits<float>::lowest();

    for (size_t i = 0; i < size; ++i)
    {
        if (buffer[i] < min_val)
        {
            min_val = buffer[i];
        }
        if (buffer[i] > max_val)
        {
            max_val = buffer[i];
        }
    }

    float range = max_val - min_val;
    for (size_t i = 0; i < size; ++i)
    {
        buffer[i] = (buffer[i] - min_val) / range;
    }
}

static const std::vector<std::string> s_kKernelNames = { "argmax" };
const std::vector<std::string> &ArgMax::GetKernelNames()
{
    return s_kKernelNames;
}

ArgMax::ArgMax(const std::string &name,
    std::shared_ptr<OpenclWrapper> openclWrapper) :
    Layer(name, openclWrapper), m_inputSize{}
{
}

void ArgMax::SetParameters(const std::vector<std::string> &elements)
{
    if (elements.size() != 3)
    {
        ALOG_GPUML("Argmax element size must be 3 represents input dimension and input channels");
    }

    auto kTotalSizeElements = 3;
    std::vector<int> parameters(kTotalSizeElements);
    int numCount = 0;
    for (int i = 0; i < kTotalSizeElements; i++)
    {
        parameters[numCount++] = std::atoi(elements[i].c_str());
    }
    std::copy(parameters.begin(), parameters.end(), m_inputSize);
}

void ArgMax::SetKernelArguments()
{
    int argCnt = 0;
    m_dimension = 2;
    m_globalSize[0] = m_inputSize[0];
    m_globalSize[1] = m_inputSize[1];
    if (m_src.size() != 1)
    {
        ALOG_GPUML("ArgMax : No src memory is created. Failed to set kernel arguments");
        return;
    }

    cl_int p = m_inputSize[0];
    cl_int q = m_inputSize[1];
    cl_int r = m_inputSize[2];
    clSetKernelArg(m_kernels[0], argCnt++, sizeof(cl_mem), &(m_src[0]->GetBuffer()));
    clSetKernelArg(m_kernels[0], argCnt++, sizeof(cl_mem), &(m_dest->GetBuffer()));
    clSetKernelArg(m_kernels[0], argCnt++, sizeof(cl_int), &p);
    clSetKernelArg(m_kernels[0], argCnt++, sizeof(cl_int), &q);
    clSetKernelArg(m_kernels[0], argCnt++, sizeof(cl_int), &r);
}

void ArgMax::CreateBuffers(const std::vector<std::shared_ptr<DataContainerOpenCLFloat>> &src)
{
    m_src = src;
    if (m_src.empty())
    {
        auto mem = std::make_shared<DataContainerOpenCLFloat>(
            std::vector{ m_inputSize[0], m_inputSize[1], m_inputSize[2] });
        mem->Allocate(m_openclWrapper->m_context);
        m_src.push_back(mem);
    }
    if (m_dest == nullptr)
    {
        auto mem = std::make_shared<DataContainerOpenCLFloat>(std::vector{ m_inputSize[0], m_inputSize[1] });
        mem->Allocate(m_openclWrapper->m_context);
        m_dest = mem;
    }
}

void ArgMax::CopyOutputBuffer(uint8_t *outBuffer, int32_t)
{
    clEnqueueReadBuffer(m_openclWrapper->m_commandQueue,
        m_dest->GetBuffer(),
        CL_TRUE,
        0,
        m_inputSize[0] * m_inputSize[1] * sizeof(uint8_t),
        outBuffer,
        0,
        0,
        0);
}

void ArgMax::GetDimension(std::vector<std::vector<uint32_t>> &dimension)
{
    dimension = { { m_inputSize[0], m_inputSize[1], 1 } };
}

void ArgMax::GetOutputTypeSizesInbyte(std::vector<uint32_t> &dimension)
{
    dimension = { sizeof(uint8_t) };
}

ArgMax::~ArgMax() {}
