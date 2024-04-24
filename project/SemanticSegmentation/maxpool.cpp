#include "maxpool.h"
#include "csvReader.h"
#include <cstdlib>
#include <random>

MaxPool::MaxPool(const std::string &name, std::shared_ptr<OpenclWrapper> openclWrapper) :
    Layer(name, openclWrapper), m_inputSize{}, m_outputSize{}, m_stride{}
{
}

static const std::vector<std::string> s_kKernelNames = { "maxpool" };
const std::vector<std::string> &MaxPool::GetKernelNames()
{
    return s_kKernelNames;
}

void MaxPool::SetParameters(const std::vector<std::string> &elements)
{
    if (elements.size() != 5)
    {
        ALOG_GPUML("MaxPool element size must be 5 represents input dimension and output channels");
    }
    else
    {
        auto kTotalSizeElements = 4;
        std::vector<int> parameters(kTotalSizeElements);
        int numCount = 0;
        for (int i = 0; i < kTotalSizeElements; i++)
        {
            parameters[numCount++] = std::atoi(elements[i].c_str());
        }

        std::copy(parameters.begin(), parameters.end() - 1, m_inputSize);
        m_stride = parameters[3];
        m_outputSize[0] = (m_inputSize[0] + m_stride - 1) / m_stride;
        m_outputSize[1] = (m_inputSize[1] + m_stride - 1) / m_stride;
        m_outputSize[2] = m_inputSize[2];
    }
}
void MaxPool::SetKernelArguments()
{
    int argCnt = 0;
    m_dimension = 3;
    m_globalSize[0] = m_outputSize[0];
    m_globalSize[1] = m_outputSize[1];
    m_globalSize[2] = m_outputSize[2] / 16;
    if (m_src.size() != 1)
    {
        ALOG_GPUML("No src memory is created. Failed to set kernel arguments");
        return;
    }
    if (m_dest.size() != 1)
    {
        ALOG_GPUML("No dest memory is created. Failed to set kernel arguments");
        return;
    }
    clSetKernelArg(m_kernels[0], argCnt++, sizeof(cl_mem), &(m_src[0]->GetBuffer()));
    clSetKernelArg(m_kernels[0], argCnt++, sizeof(cl_mem), &(m_dest[0]->GetBuffer()));
    clSetKernelArg(m_kernels[0], argCnt++, sizeof(uint32_t), &m_inputSize[0]);
    clSetKernelArg(m_kernels[0], argCnt++, sizeof(uint32_t), &m_inputSize[1]);
    clSetKernelArg(m_kernels[0], argCnt++, sizeof(uint32_t), &m_inputSize[2]);
    clSetKernelArg(m_kernels[0], argCnt++, sizeof(uint32_t), &m_stride);
}
void MaxPool::CreateBuffers(const std::vector<std::shared_ptr<DataContainerOpenCLFloat>> &src)
{
    m_src = src;
    if (m_src.empty())
    {
        auto mem = std::make_shared<DataContainerOpenCLFloat>(
            std::vector{ m_inputSize[0], m_inputSize[1], m_inputSize[2] });
        mem->Allocate(m_openclWrapper->m_context);
        m_src.push_back(mem);
    }
    if (m_dest.empty())
    {
        auto mem = std::make_shared<DataContainerOpenCLFloat>(
            std::vector{ m_outputSize[0], m_outputSize[1], m_inputSize[2] });
        mem->Allocate(m_openclWrapper->m_context);
        m_dest.push_back(mem);
    }
}

void MaxPool::FillLayerInputFromFile(const std::filesystem::path &inputPath)
{
    auto fullPath = inputPath / (m_name + "_in.npy");
    m_src[0]->LoadFromFile(fullPath, m_openclWrapper->m_commandQueue);
}

MaxPool::~MaxPool() {}
