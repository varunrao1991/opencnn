#include "depthwiseConvolutionLayer.h"
#include "bmpreader.h"
#include "numpy.hpp"
#include <fstream>

DepthwiseConvolutionLayer::DepthwiseConvolutionLayer(
    const std::string &name, std::shared_ptr<OpenclWrapper> openclWrapper) :
    Layer(name, openclWrapper), m_inputSize{}, m_filterSize{}, m_stride{}, m_outputSize{}
{
}

static const std::vector<std::string> s_kKernelNames = { "depthwise" };
const std::vector<std::string> &DepthwiseConvolutionLayer::GetKernelNames()
{
    return s_kKernelNames;
}

void DepthwiseConvolutionLayer::SetParameters(const std::vector<std::string> &elements)
{
    if (elements.size() != 8)
    {
        ALOG_GPUML("DepthwiseConvolutionLayer element size must be 8 represents input dimension and output channels");
        return;
    }
    if (elements[5] == elements[2] && elements[0] == elements[1])
    {
        auto kTotalSizeElements = 7;
        std::vector<int> parameters(kTotalSizeElements);
        int numCount = 0;
        for (int i = 0; i < kTotalSizeElements; i++)
        {
            parameters[numCount++] = std::atoi(elements[i].c_str());
        }
        std::copy(parameters.begin(), parameters.begin() + 3, m_filterSize);
        std::copy(parameters.begin() + 3, parameters.begin() + 7, m_inputSize);
        m_stride = parameters[elements.size() - 2];
        m_outputSize[0] = (m_inputSize[0] + m_stride - 1) / m_stride;
        m_outputSize[1] = (m_inputSize[1] + m_stride - 1) / m_stride;
        m_outputSize[2] = m_filterSize[2];
    }
    else
    {
        ALOG_GPUML("DepthwiseConvolutionLayer filter dimension and input dimension mismatch '%s'", m_name.c_str());
    }
}

void DepthwiseConvolutionLayer::SetKernelArguments()
{
    int argCnt = 0;
    m_dimension = 1;

    m_globalSize[0] = m_outputSize[0] * m_outputSize[1] * m_outputSize[2];

    if (m_src.size() != 2)
    {
        ALOG_GPUML("DepthwiseConvolutionLayer : No src memory is created. Failed to set kernel arguments");
        return;
    }

    clSetKernelArg(m_kernels[0], argCnt++, sizeof(cl_mem), &(m_src[0]->GetBuffer()));
    clSetKernelArg(m_kernels[0], argCnt++, sizeof(cl_mem), &(m_src[1]->GetBuffer()));
    clSetKernelArg(m_kernels[0], argCnt++, sizeof(cl_mem), &(m_dest->GetBuffer()));
    clSetKernelArg(m_kernels[0], argCnt++, sizeof(uint32_t), &m_inputSize[0]);
    clSetKernelArg(m_kernels[0], argCnt++, sizeof(uint32_t), &m_inputSize[1]);
    clSetKernelArg(m_kernels[0], argCnt++, sizeof(uint32_t), &m_inputSize[2]);
    clSetKernelArg(m_kernels[0], argCnt++, sizeof(uint32_t), &m_stride);
}

void DepthwiseConvolutionLayer::CreateBuffers(const std::vector<std::shared_ptr<DataContainerOpenCLFloat>> &src)
{
    m_src = src;
    if (m_src.empty())
    {
        auto mem = std::make_shared<DataContainerOpenCLFloat>(
            std::vector{ m_filterSize[0], m_filterSize[1], m_filterSize[2] });
        mem->Allocate(m_openclWrapper->m_context);
        m_src.push_back(mem);
        mem = std::make_shared<DataContainerOpenCLFloat>(std::vector{ m_inputSize[0], m_inputSize[1], m_inputSize[2] });
        mem->Allocate(m_openclWrapper->m_context);
        m_src.push_back(mem);
    }
    else if (m_src.size() == 1)
    {
        auto mem = std::make_shared<DataContainerOpenCLFloat>(
            std::vector{ m_filterSize[0], m_filterSize[1], m_filterSize[2] });
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
            std::vector{ m_outputSize[0], m_outputSize[1], m_filterSize[2] });
        mem->Allocate(m_openclWrapper->m_context);
        m_dest = mem;
    }
}

void DepthwiseConvolutionLayer::FillLayerInputFromFile(const std::filesystem::path &inputPath)
{
    auto fullPath = inputPath / (m_name + "_in.npy");
    m_src[1]->LoadFromFile(fullPath, m_openclWrapper->m_commandQueue);
}

void DepthwiseConvolutionLayer::FillLayerConstants(const std::filesystem::path &inputPath)
{
    auto fullPath = inputPath / (m_name + "_weights.npy");
    m_src[0]->LoadFromFile(fullPath, m_openclWrapper->m_commandQueue);
}

void DepthwiseConvolutionLayer::writeInputBuffers(const std::filesystem::path &outputPath)
{
    auto fileName = outputPath / (m_name + "_0.bin");
    m_src[0]->ExportDataInBin(fileName, m_openclWrapper->m_commandQueue);
    fileName = outputPath / (m_name + "_1.bin");
    m_src[1]->ExportDataInBin(fileName, m_openclWrapper->m_commandQueue);
}

void DepthwiseConvolutionLayer::writeOutputBuffers(const std::filesystem::path &outputPath)
{
    auto fileName = outputPath / (m_name + "_out.bin");
    m_dest->ExportDataInBin(fileName, m_openclWrapper->m_commandQueue);
}

DepthwiseConvolutionLayer::~DepthwiseConvolutionLayer() {}
