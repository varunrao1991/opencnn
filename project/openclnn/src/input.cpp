#include "input.h"
#include "bmpreader.h"
#include "numpy.hpp"

#include <numeric>

InputLayer::InputLayer(const std::string &name, std::shared_ptr<OpenclWrapper> openclWrapper) :
    Layer(name, openclWrapper), m_inputSize{}
{
}

static const std::vector<std::string> s_kKernelNames = { "preprocess" };

const std::vector<std::string> &InputLayer::GetKernelNames()
{
    return s_kKernelNames;
}

void InputLayer::SetParameters(const std::vector<std::string> &elements)
{
    const int32_t kTotalSizeElements = 3;
    if (elements.size() != kTotalSizeElements)
    {
        ALOG_GPUML("InputLayer element size must be 3 represents input dimension, channels");
        return;
    }
    std::vector<int> parameters(kTotalSizeElements);
    int numCount = 0;
    for (int i = 0; i < kTotalSizeElements; i++)
    {
        parameters[numCount++] = std::atoi(elements[i].c_str());
    }
    std::copy(parameters.begin(), parameters.begin() + 3, m_inputSize);
    ALOG_GPUML("[InputLayer] element [%d, %d, %d]", m_inputSize[0], m_inputSize[1], m_inputSize[2]);
}

void InputLayer::GetDimension(uint32_t &width, uint32_t &height)
{
    width = m_inputSize[0];
    height = m_inputSize[1];
}

void InputLayer::SetKernelArguments()
{
    int argCnt = 0;
    m_dimension = 2;
    m_globalSize[0] = static_cast<size_t>(m_inputSize[0]) * m_inputSize[1];
    m_globalSize[1] = m_inputSize[2];
    if (m_src.size() != 1)
    {
        ALOG_GPUML("No src memory is created. Failed to set kernel arguments");
        return;
    }

    clSetKernelArg(m_kernels[0], argCnt++, sizeof(cl_mem), &(m_src[0]->GetBuffer()));
    clSetKernelArg(m_kernels[0], argCnt++, sizeof(cl_mem), &(m_dest->GetBuffer()));
    clSetKernelArg(m_kernels[0], argCnt++, sizeof(uint32_t), &m_inputSize[2]);
}

void InputLayer::CreateBuffers(const std::vector<std::shared_ptr<DataContainerOpenCLFloat>> &src)
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
        auto mem = std::make_shared<DataContainerOpenCLFloat>(
            std::vector{ m_inputSize[0], m_inputSize[1], m_inputSize[2] });
        mem->Allocate(m_openclWrapper->m_context);
        m_dest = mem;
    }
}

void InputLayer::FillInputFromBuffer(float *dataIn)
{
    m_src[0]->FillData(m_openclWrapper->m_commandQueue, dataIn);
}

void InputLayer::FillLayerInputFromFile(const std::filesystem::path &inputPath)
{
    auto filePath = inputPath / ("in_" + m_name + ".npy");
    m_src[0]->LoadFromFile(filePath, m_openclWrapper->m_commandQueue);
}

void InputLayer::GetDimension(std::vector<std::vector<uint32_t>> &dimension)
{
    dimension = { { m_inputSize[0], m_inputSize[1], m_inputSize[2] } };
}

InputLayer::~InputLayer() {}
