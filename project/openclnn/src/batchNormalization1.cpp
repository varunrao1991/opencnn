#include "batchNormalization1.h"
#include "bmpreader.h"
#include "csvReader.h"
#include "numpy.hpp"

#include <iomanip>

BatchNormalization1::BatchNormalization1(const std::string &name, std::shared_ptr<OpenclWrapper> openclWrapper) :
    Layer(name, openclWrapper), m_inputSize{}
{
}

static const std::vector<std::string> s_kKernelNames = { "batchnormalization1" };

const std::vector<std::string> &BatchNormalization1::GetKernelNames()
{
    return s_kKernelNames;
}

void BatchNormalization1::SetParameters(const std::vector<std::string> &elements)
{
    if (elements.size() != 3)
    {
        ALOG_GPUML("BatchNormalization1 element size must be 3 represents input dimension and output channels");
        return;
    }
    else
    {
        auto kTotalSizeElements = 3;
        std::vector<int> parameters(kTotalSizeElements);
        int numCount = 0;
        for (int i = 0; i < kTotalSizeElements; i++)
        {
            parameters[numCount++] = std::atoi(elements[i].c_str());
        }
        std::copy(parameters.begin(), parameters.end(), m_inputSize);
    }
}

void BatchNormalization1::SetKernelArguments()
{
    int argCnt = 0;
    m_dimension = 3;
    m_globalSize[0] = m_inputSize[0];
    m_globalSize[1] = m_inputSize[1];

    m_globalSize[2] = m_inputSize[2];

    if (m_src.size() != 3)
    {
        ALOG_GPUML("BatchNormalization1 : No src memory is created. Failed to set kernel arguments");
        return;
    }
    if (m_dest.size() != 1)
    {
        ALOG_GPUML("No dest memory is created. Failed to set kernel arguments");
        return;
    }

    clSetKernelArg(m_kernels[0], argCnt++, sizeof(cl_mem), &(m_src[0]->GetBuffer()));
    clSetKernelArg(m_kernels[0], argCnt++, sizeof(cl_mem), &(m_src[1]->GetBuffer()));
    clSetKernelArg(m_kernels[0], argCnt++, sizeof(cl_mem), &(m_src[2]->GetBuffer()));
    clSetKernelArg(m_kernels[0], argCnt++, sizeof(cl_mem), &(m_dest[0]->GetBuffer()));
    clSetKernelArg(m_kernels[0], argCnt++, sizeof(uint32_t), &m_inputSize[0]);
    clSetKernelArg(m_kernels[0], argCnt++, sizeof(uint32_t), &m_inputSize[1]);
    clSetKernelArg(m_kernels[0], argCnt++, sizeof(uint32_t), &m_inputSize[2]);
}

void BatchNormalization1::CreateBuffers(const std::vector<std::shared_ptr<DataContainerOpenCLFloat>> &src)
{
    m_src = src;
    if (m_src.empty())
    {
        auto mem = std::make_shared<DataContainerOpenCLFloat>(m_inputSize[2]);
        mem->Allocate(m_openclWrapper->m_context);
        m_src.push_back(mem);
        mem = std::make_shared<DataContainerOpenCLFloat>(m_inputSize[2]);
        mem->Allocate(m_openclWrapper->m_context);
        m_src.push_back(mem);
        mem = std::make_shared<DataContainerOpenCLFloat>(
            std::vector{ m_inputSize[0], m_inputSize[1], m_inputSize[2] });
        mem->Allocate(m_openclWrapper->m_context);
        m_src.push_back(mem);
    }
    else if (m_src.size() == 1)
    {
        auto mem = std::make_shared<DataContainerOpenCLFloat>(m_inputSize[2]);
        mem->Allocate(m_openclWrapper->m_context);
        m_src.insert(m_src.begin(), mem);
        mem = std::make_shared<DataContainerOpenCLFloat>(m_inputSize[2]);
        mem->Allocate(m_openclWrapper->m_context);
        m_src.insert(m_src.begin(), mem);
    }
    else
    {
        ALOG_GPUML("Buffer passed is beyond the requirement");
    }

    if (m_dest.empty())
    {
        auto mem = std::make_shared<DataContainerOpenCLFloat>(
            std::vector{ m_inputSize[0], m_inputSize[1], m_inputSize[2] });
        mem->Allocate(m_openclWrapper->m_context);
        m_dest.push_back(mem);
    }
}

void BatchNormalization1::FillLayerInputFromFile(const std::filesystem::path &inputPath)
{
    auto fullPath = inputPath / (m_name + "_in.npy");
    m_src[2]->LoadFromFile(fullPath, m_openclWrapper->m_commandQueue);
}

void BatchNormalization1::FillLayerConstants(const std::filesystem::path &inputPath)
{
    auto fullPath = inputPath / (m_name + "_mean.npy");
    m_src[0]->LoadFromFile(fullPath, m_openclWrapper->m_commandQueue);

    fullPath = inputPath / (m_name + "_variance.npy");
    m_src[1]->LoadFromFile(fullPath, m_openclWrapper->m_commandQueue);
}

BatchNormalization1::~BatchNormalization1() {}
