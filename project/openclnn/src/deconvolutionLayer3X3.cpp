#include "deconvolutionLayer3X3.h"
#include "numpy.hpp"
#include <fstream>

DeconvolutionLayer3X3::DeconvolutionLayer3X3(const std::string &name, std::shared_ptr<OpenclWrapper> openclWrapper) :
    Layer(name, openclWrapper), m_inputSize{}, m_filterSize{}, m_runCount{}, m_outputSize{}
{
}

static const std::vector<std::string> s_kKernelNames = { "deconvolution3x3" };
const std::vector<std::string> &DeconvolutionLayer3X3::GetKernelNames()
{
    return s_kKernelNames;
}

void DeconvolutionLayer3X3::SetParameters(const std::vector<std::string> &elements)
{
    if (elements.size() != 7)
    {
        ALOG_GPUML("DeconvolutionLayer3X3 element size must be 7 represents input dimension and output channels");
        return;
    }
    if (elements[6] == elements[2])
    {
        auto kTotalSizeElements = 7;
        std::vector<int> parameters(kTotalSizeElements);
        int numCount = 0;
        for (int i = 0; i < kTotalSizeElements; i++)
        {
            parameters[numCount++] = std::atoi(elements[i].c_str());
        }
        std::copy(parameters.begin(), parameters.begin() + 4, m_filterSize);
        std::copy(parameters.begin() + 4, parameters.end(), m_inputSize);

        m_outputSize[0] = 2 * m_inputSize[0];
        m_outputSize[1] = 2 * m_inputSize[1];
        m_outputSize[2] = m_filterSize[2];
        m_cpuMem = std::make_unique<float[]>(m_outputSize[0] * m_outputSize[1] * m_outputSize[2]);
        memset(m_cpuMem.get(), 0, m_outputSize[0] * m_outputSize[1] * m_outputSize[2] * sizeof(float));
    }
    else
        ALOG_GPUML("DeconvolutionLayer3X3 filter dimension and input dimension mismatch");
}

void DeconvolutionLayer3X3::SetKernelArguments()
{
    int argCnt = 0;
    m_dimension = 3;
    m_globalSize[0] = m_inputSize[0] / 2;
    m_globalSize[1] = m_inputSize[1] / 2;
    m_globalSize[2] = m_outputSize[2] / 16;
    if (m_src.size() != 2)
    {
        ALOG_GPUML("DeconvolutionLayer3X3 : No src memory is created. Failed to set kernel arguments");
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
    clSetKernelArg(m_kernels[0], argCnt++, sizeof(uint32_t), &m_outputSize[2]);
    clSetKernelArg(m_kernels[0], argCnt++, sizeof(cl_int), &m_runCount);
}

void DeconvolutionLayer3X3::CreateBuffers(const std::vector<std::shared_ptr<DataContainerOpenCLFloat>> &src)
{
    m_src = src;
    if (m_src.empty())
    {
        auto mem = std::make_shared<DataContainerOpenCLFloat>(
            std::vector{ m_filterSize[0], m_filterSize[1], m_filterSize[2], m_filterSize[3] });
        mem->Allocate(m_openclWrapper->m_context);
        m_src.push_back(mem);
        mem = std::make_shared<DataContainerOpenCLFloat>(std::vector{ m_inputSize[0], m_inputSize[1], m_inputSize[2] });
        mem->Allocate(m_openclWrapper->m_context);
        m_src.push_back(mem);
    }
    else if (m_src.size() == 1)
    {
        auto mem = std::make_shared<DataContainerOpenCLFloat>(
            std::vector{ m_filterSize[0], m_filterSize[1], m_filterSize[2], m_filterSize[3] });
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
            std::vector{ 2 * m_inputSize[0], 2 * m_inputSize[1], m_filterSize[3] });
        mem->Allocate(m_openclWrapper->m_context);
        m_dest.push_back(mem);
    }
}

void DeconvolutionLayer3X3::FillLayerInputFromFile(const std::filesystem::path &inputPath)
{
    auto fullPath = inputPath / (m_name + "filter.npy");
    m_src[0]->LoadFromFile(fullPath, m_openclWrapper->m_commandQueue);
    m_dest[0]->ResetData(m_openclWrapper->m_commandQueue);
}

void DeconvolutionLayer3X3::FillLayerConstants(const std::filesystem::path &inputPath)
{
    auto fullPath = inputPath / (m_name + "_weights.npy");
    m_src[0]->LoadFromFile(fullPath, m_openclWrapper->m_commandQueue);
    m_dest[0]->ResetData(m_openclWrapper->m_commandQueue);
}

void DeconvolutionLayer3X3::EnqueueKernel()
{
    if (m_cpuMem)
    {
        m_dest[0]->FillData(m_openclWrapper->m_commandQueue, m_cpuMem.get());
    }
    m_runCount = 0;
    for (int i = 0; i < 4; i++)
    {
        SetKernelArguments();
        Layer::EnqueueKernel();
        m_runCount++;
    }
}

DeconvolutionLayer3X3::~DeconvolutionLayer3X3() {}
