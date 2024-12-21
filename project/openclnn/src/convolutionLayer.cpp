#include "convolutionLayer.h"
#include "bmpreader.h"
#include "csvReader.h"
#include "numpy.hpp"

#include <iomanip>

ConvolutionLayer::ConvolutionLayer(const std::string &name, std::shared_ptr<OpenclWrapper> openclWrapper) :
    Layer(name, openclWrapper), m_inputSize{}, m_filterSize{}, m_stride{}, m_enableRelu{ false }, m_outputSize{}
{
}

static const std::vector<std::string> s_kKernelNames = { "convolution" };
const std::vector<std::string> &ConvolutionLayer::GetKernelNames()
{
    return s_kKernelNames;
}

void ConvolutionLayer::SetParameters(const std::vector<std::string> &elements)
{
    if (elements.size() != 11)
    {
        ALOG_GPUML("ConvolutionLayer element size must be 11 provided %d", elements.size());
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
    }
    else
    {
        ALOG_GPUML("ConvolutionLayer filter dimension and input dimension mismatch '%s'", m_name.c_str());
    }
}

void ConvolutionLayer::SetKernelArguments()
{
    int argCnt = 0;
    m_dimension = 3;
    m_globalSize[0] = m_outputSize[0];
    m_globalSize[1] = m_outputSize[1];
    m_globalSize[2] = (m_filterSize[3] + 15) / 16;

    if (m_src.size() != 2)
    {
        ALOG_GPUML("ConvolutionLayer : No src memory is created. Failed to set kernel arguments");
        return;
    }

    clSetKernelArg(m_kernels[0], argCnt++, sizeof(cl_mem), &(m_src[0]->GetBuffer()));
    clSetKernelArg(m_kernels[0], argCnt++, sizeof(cl_mem), &(m_src[1]->GetBuffer()));
    clSetKernelArg(m_kernels[0], argCnt++, sizeof(cl_mem), &(m_dest->GetBuffer()));
    clSetKernelArg(m_kernels[0], argCnt++, sizeof(uint32_t), &m_inputSize[0]);
    clSetKernelArg(m_kernels[0], argCnt++, sizeof(uint32_t), &m_inputSize[1]);
    clSetKernelArg(m_kernels[0], argCnt++, sizeof(uint32_t), &m_filterSize[2]);
    clSetKernelArg(m_kernels[0], argCnt++, sizeof(uint32_t), &m_filterSize[3]);
    clSetKernelArg(m_kernels[0], argCnt++, sizeof(uint32_t), &m_stride);
    clSetKernelArg(m_kernels[0], argCnt++, sizeof(cl_char), &m_enableRelu);
}
void ConvolutionLayer::CreateBuffers(const std::vector<std::shared_ptr<DataContainerOpenCLFloat>> &src)
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
    if (m_dest == nullptr)
    {
        auto mem =
            std::make_shared<DataContainerOpenCLFloat>(std::vector{ m_inputSize[0], m_inputSize[1], m_filterSize[3] });
        mem->Allocate(m_openclWrapper->m_context);
        m_dest = mem;
    }
}

void ConvolutionLayer::FillLayerInputFromFile(const std::filesystem::path &inputPath)
{
    auto fullPath = inputPath / (m_name + "_in.npy");
    m_src[1]->LoadFromFile(fullPath, m_openclWrapper->m_commandQueue);
}

void ConvolutionLayer::FillLayerConstants(const std::filesystem::path &inputPath)
{
    auto fullPath = inputPath / (m_name + "_weights.npy");
    m_src[0]->LoadFromFile(fullPath, m_openclWrapper->m_commandQueue);
}

void ConvolutionLayer::EnableReluActivation()
{
    m_enableRelu = true;
}

ConvolutionLayer::~ConvolutionLayer() {}
