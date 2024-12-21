#include "convolutionLayerBias.h"
#include "bmpreader.h"
#include "csvReader.h"
#include "numpy.hpp"

#include <iomanip>

ConvolutionLayerBias::ConvolutionLayerBias(const std::string &name, std::shared_ptr<OpenclWrapper> openclWrapper) :
    Layer(name, openclWrapper),
    m_inputSize{},
    m_enableRelu{ false },
    m_stride{ 0 },
    m_filterSize{},
    m_enableTranspose{ false }
{
}

static const std::vector<std::string> s_kKernelNames = { "convolutionBias", "convolutionBiasTransposed" };
const std::vector<std::string> &ConvolutionLayerBias::GetKernelNames()
{
    return s_kKernelNames;
}

void ConvolutionLayerBias::SetParameters(const std::vector<std::string> &elements)
{
    if (elements.size() != 11 && elements.size() != 12)
    {
        ALOG_GPUML("ConvolutionLayerBias element size must be 11 provided %zd", elements.size());
        return;
    }
    if (elements[7] == elements[3])
    {
        auto kTotalSizeElements = 8;
        std::vector<int> parameters(kTotalSizeElements);
        int numCount = 0;
        for (int i = 0; i < kTotalSizeElements; i++)
        {
            parameters[numCount++] = std::atoi(elements[i + 1].c_str());
        }
        std::copy(parameters.begin(), parameters.begin() + 4, m_filterSize);
        std::copy(parameters.begin() + 4, parameters.begin() + 7, m_inputSize);
        m_stride = parameters[parameters.size() - 1];
        m_outputSize[0] = (m_inputSize[0] + m_stride - 1) / m_stride;
        m_outputSize[1] = (m_inputSize[1] + m_stride - 1) / m_stride;
        m_outputSize[2] = m_filterSize[3];
        if (elements[0] == "relu")
        {
            EnableReluActivation();
        }
        if (elements.size() > 11)
        {
            m_enableTranspose = elements[11] == "True" ? true : false;
        }
    }
    else
    {
        ALOG_GPUML("ConvolutionLayerBias filter dimension and input dimension mismatch '%s'", m_name.c_str());
    }
}

void ConvolutionLayerBias::EnableReluActivation()
{
    m_enableRelu = true;
}

void ConvolutionLayerBias::EnqueueKernel()
{
    SetKernelArguments();
    auto &kernel = m_enableTranspose ? m_kernels[1] : m_kernels[0];

    cl_int err = clEnqueueNDRangeKernel(
        m_openclWrapper->m_commandQueue, kernel, m_dimension, m_globalOffset, m_globalSize, 0, 0, 0, nullptr);
    ALOG_GPUML_CHECK_ERROR1(err, __LINE__, "clEnqueueNDRangeKernel");
}

void ConvolutionLayerBias::SetKernelArguments()
{
    int argCnt = 0;
    m_dimension = 3;
    m_globalSize[0] = m_outputSize[0];
    m_globalSize[1] = m_outputSize[1];

    bool kDivisibility = m_filterSize[3] % 16 == 0;
    if (m_inputSize[0] == 19 || m_inputSize[0] == 1) // TODO: remove specialization
    {
        kDivisibility = false;
    }
    m_globalSize[2] = kDivisibility ? m_filterSize[3] / 16 : m_filterSize[3] / 2;

    if (m_src.size() != 3)
    {
        ALOG_GPUML("ConvolutionLayerBias : No src memory is created. Failed to set kernel arguments");
        return;
    }

    auto &kernel = m_enableTranspose ? m_kernels[1] : m_kernels[0];

    clSetKernelArg(kernel, argCnt++, sizeof(cl_mem), &(m_src[0]->GetBuffer()));
    clSetKernelArg(kernel, argCnt++, sizeof(cl_mem), &(m_src[1]->GetBuffer()));
    clSetKernelArg(kernel, argCnt++, sizeof(cl_mem), &(m_src[2]->GetBuffer()));
    clSetKernelArg(kernel, argCnt++, sizeof(cl_mem), &(m_dest->GetBuffer()));
    clSetKernelArg(kernel, argCnt++, sizeof(uint32_t), &m_inputSize[0]);
    clSetKernelArg(kernel, argCnt++, sizeof(uint32_t), &m_inputSize[1]);
    clSetKernelArg(kernel, argCnt++, sizeof(uint32_t), &m_filterSize[2]);
    clSetKernelArg(kernel, argCnt++, sizeof(uint32_t), &m_filterSize[3]);
    clSetKernelArg(kernel, argCnt++, sizeof(uint32_t), &m_stride);
    clSetKernelArg(kernel, argCnt++, sizeof(cl_char), &kDivisibility);
    clSetKernelArg(kernel, argCnt++, sizeof(cl_char), &m_enableRelu);
}

void ConvolutionLayerBias::CreateBuffers(const std::vector<std::shared_ptr<DataContainerOpenCLFloat>> &src)
{
    m_src = src;
    if (m_src.empty())
    {
        auto mem = std::make_shared<DataContainerOpenCLFloat>(
            std::vector{ m_filterSize[0], m_filterSize[1], m_filterSize[2], m_filterSize[3] });
        mem->Allocate(m_openclWrapper->m_context);
        m_src.push_back(mem);
        mem = std::make_shared<DataContainerOpenCLFloat>(std::vector{ m_filterSize[3] });
        mem->Allocate(m_openclWrapper->m_context);
        m_src.push_back(mem);
        mem = std::make_shared<DataContainerOpenCLFloat>(std::vector{ m_inputSize[0], m_inputSize[1], m_inputSize[2] });
        mem->Allocate(m_openclWrapper->m_context);
        m_src.push_back(mem);
    }
    else if (m_src.size() == 1)
    {
        auto mem = std::make_shared<DataContainerOpenCLFloat>(m_filterSize[3]);
        mem->Allocate(m_openclWrapper->m_context);
        m_src.insert(m_src.begin(), mem);
        mem = std::make_shared<DataContainerOpenCLFloat>(
            std::vector{ m_filterSize[0], m_filterSize[1], m_filterSize[2], m_filterSize[3] });
        mem->Allocate(m_openclWrapper->m_context);
        m_src.insert(m_src.begin(), mem);
    }
    else
    {
        ALOG_GPUML("Buffer passed is beyond the requirement");
    }
    if (m_dest == nullptr)
    {
        auto mem = std::make_shared<DataContainerOpenCLFloat>(
            std::vector{ m_outputSize[0], m_outputSize[1], m_outputSize[2] });
        mem->Allocate(m_openclWrapper->m_context);
        m_dest = mem;
    }
}

void ConvolutionLayerBias::FillLayerInputFromFile(const std::filesystem::path &inputPath)
{
    auto fullPath = inputPath / (m_name + "_in.npy");
    m_src[2]->LoadFromFile(fullPath, m_openclWrapper->m_commandQueue);
}

void ConvolutionLayerBias::FillLayerConstants(const std::filesystem::path &inputPath)
{
    auto fullPath = inputPath / (m_name + "_weights.npy");
    m_src[0]->LoadFromFile(fullPath, m_openclWrapper->m_commandQueue);

    fullPath = inputPath / (m_name + "_bias.npy");
    m_src[1]->LoadFromFile(fullPath, m_openclWrapper->m_commandQueue);
}

ConvolutionLayerBias::~ConvolutionLayerBias() {}
