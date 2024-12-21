#include "padding.h"
#include "bmpreader.h"
#include "numpy.hpp"
#include <fstream>

PaddingLayer::PaddingLayer(const std::string &name, std::shared_ptr<OpenclWrapper> openclWrapper) :
    Layer(name, openclWrapper), m_inputSize{}, m_outputSize{}, m_padding{}
{
}

static const std::vector<std::string> s_kKernelNames = { "padding" };
const std::vector<std::string> &PaddingLayer::GetKernelNames()
{
    return s_kKernelNames;
}

void PaddingLayer::SetParameters(const std::vector<std::string> &elements)
{
    if (elements.size() != 7)
    {
        ALOG_GPUML("PaddingLayer element size must be 4 represents input dimension");
        return;
    }

    auto kTotalSizeElements = 7;
    std::vector<int> parameters(kTotalSizeElements);
    int numCount = 0;
    for (int i = 0; i < kTotalSizeElements; i++)
    {
        parameters[numCount++] = std::atoi(elements[i].c_str());
    }
    std::copy(parameters.begin(), parameters.begin() + 3, m_inputSize);
    std::copy(parameters.begin() + 3, parameters.end(), m_padding);
    setPadding();
}

void PaddingLayer::SetKernelArguments()
{
    int argCnt = 0;
    m_dimension = 3;
    m_globalSize[0] = m_outputSize[0];
    m_globalSize[1] = m_outputSize[1];
    m_globalSize[2] = m_outputSize[2];
    float paddingValue = 0.0f;

    if (m_src.size() != 1)
    {
        ALOG_GPUML("PaddingLayer : No src memory is created. Failed to set kernel arguments");
        return;
    }


    clSetKernelArg(m_kernels[0], argCnt++, sizeof(cl_mem), &(m_src[0]->GetBuffer()));
    clSetKernelArg(m_kernels[0], argCnt++, sizeof(cl_mem), &(m_dest->GetBuffer()));
    clSetKernelArg(m_kernels[0], argCnt++, sizeof(uint32_t), &m_inputSize[0]);
    clSetKernelArg(m_kernels[0], argCnt++, sizeof(uint32_t), &m_inputSize[1]);
    clSetKernelArg(m_kernels[0], argCnt++, sizeof(uint32_t), &m_inputSize[2]);
    clSetKernelArg(m_kernels[0], argCnt++, sizeof(uint32_t), &m_padding[0]);
    clSetKernelArg(m_kernels[0], argCnt++, sizeof(uint32_t), &m_padding[1]);
    clSetKernelArg(m_kernels[0], argCnt++, sizeof(uint32_t), &m_padding[2]);
    clSetKernelArg(m_kernels[0], argCnt++, sizeof(uint32_t), &m_padding[3]);
    clSetKernelArg(m_kernels[0], argCnt++, sizeof(cl_float), &paddingValue);
}

void PaddingLayer::setPadding()
{
    m_outputSize[0] = m_inputSize[0] + m_padding[0] + m_padding[1];
    m_outputSize[1] = m_inputSize[1] + m_padding[2] + m_padding[3];
    m_outputSize[2] = m_inputSize[2];
}

void PaddingLayer::CreateBuffers(const std::vector<std::shared_ptr<DataContainerOpenCLFloat>> &src)
{
    m_src = src;
    if (m_src.empty())
    {
        auto mem = std::make_shared<DataContainerOpenCLFloat>(
            std::vector{ m_inputSize[0], m_inputSize[1], m_inputSize[2] });
        mem->Allocate(m_openclWrapper->m_context);
        m_src.push_back(mem);
    }
    else if (m_src.size() != 1)
    {
        ALOG_GPUML("[PaddingLayer] Buffer passed is beyond the requirement");
    }
    if (m_dest == nullptr)
    {
        auto mem = std::make_shared<DataContainerOpenCLFloat>(
            std::vector{ m_outputSize[0], m_outputSize[1], m_outputSize[2] });
        mem->Allocate(m_openclWrapper->m_context);
        m_dest = mem;
    }
}

void PaddingLayer::FillLayerInputFromFile(const std::filesystem::path &inputPath)
{
    auto fullPath = inputPath / (m_name + "_in.npy");
    m_src[0]->LoadFromFile(fullPath, m_openclWrapper->m_commandQueue);
}

void PaddingLayer::writeOutputBuffers(const std::filesystem::path &outputPath)
{
    auto fileName = outputPath / (m_name + "_out.bin");
    m_dest->ExportDataInBin(fileName, m_openclWrapper->m_commandQueue);
}

PaddingLayer::~PaddingLayer() {}
