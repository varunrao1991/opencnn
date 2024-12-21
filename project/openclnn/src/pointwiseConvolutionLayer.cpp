#include "PointwiseConvolutionLayer.h"
#include "bmpreader.h"
#include "numpy.hpp"

PointwiseConvolutionLayer::PointwiseConvolutionLayer(
    const std::string &name, std::shared_ptr<OpenclWrapper> openclWrapper) :
    Layer(name, openclWrapper), m_inputSize{}, m_enableRelu{ false }, m_filterSize{}
{
}

static const std::vector<std::string> s_kKernelNames = { "pointwise" };
const std::vector<std::string> &PointwiseConvolutionLayer::GetKernelNames()
{
    return s_kKernelNames;
}

void PointwiseConvolutionLayer::SetParameters(const std::vector<std::string> &elements)
{
    if (elements.size() != 11)
    {
        ALOG_GPUML("PointwiseConvolutionLayer element size must be 11 provided %zd", elements.size());
        return;
    }
    if (elements[1] == elements[2] && elements[1] == "1" && elements[elements.size() - 3] == "1" &&
        elements[elements.size() - 2] == "False")
    {
        auto kTotalSizeElements = 8;
        std::vector<int> parameters(kTotalSizeElements);
        int numCount = 0;
        for (int i = 0; i < kTotalSizeElements; i++)
        {
            parameters[numCount++] = std::atoi(elements[i + 1].c_str());
        }
        std::copy(parameters.begin() + 2, parameters.begin() + 4, m_filterSize);
        std::copy(parameters.begin() + 4, parameters.begin() + 7, m_inputSize);
        if (elements[0] == "relu")
        {
            EnableReluActivation();
        }
    }
    else
    {
        ALOG_GPUML("PointwiseConvolutionLayer filter dimension and input dimension mismatch '%s'", m_name.c_str());
    }
}

void PointwiseConvolutionLayer::EnableReluActivation()
{
    m_enableRelu = true;
}

void PointwiseConvolutionLayer::SetKernelArguments()
{
    int argCnt = 0;
    m_dimension = 2;
    m_globalSize[0] = static_cast<size_t>(m_inputSize[0]) * m_inputSize[1];
    m_globalSize[1] = m_filterSize[1] / 2;
    if (m_src.size() != 2)
    {
        ALOG_GPUML("PointwiseConvolutionLayer : No src memory is created. Failed to set kernel arguments");
        return;
    }

    clSetKernelArg(m_kernels[0], argCnt++, sizeof(cl_mem), &(m_src[0]->GetBuffer()));
    clSetKernelArg(m_kernels[0], argCnt++, sizeof(cl_mem), &(m_src[1]->GetBuffer()));
    clSetKernelArg(m_kernels[0], argCnt++, sizeof(cl_mem), &(m_dest->GetBuffer()));
    clSetKernelArg(m_kernels[0], argCnt++, sizeof(uint32_t), &m_inputSize[0]);
    clSetKernelArg(m_kernels[0], argCnt++, sizeof(uint32_t), &m_inputSize[1]);
    clSetKernelArg(m_kernels[0], argCnt++, sizeof(uint32_t), &m_filterSize[0]);
    clSetKernelArg(m_kernels[0], argCnt++, sizeof(uint32_t), &m_filterSize[1]);
    clSetKernelArg(m_kernels[0], argCnt++, sizeof(cl_char), &m_enableRelu);
}

void PointwiseConvolutionLayer::CreateBuffers(const std::vector<std::shared_ptr<DataContainerOpenCLFloat>> &src)
{
    m_src = src;
    if (m_src.empty())
    {
        auto mem =
            std::make_shared<DataContainerOpenCLFloat>(std::vector{ m_filterSize[0], m_filterSize[1] });
        mem->Allocate(m_openclWrapper->m_context);
        m_src.push_back(mem);
        mem = std::make_shared<DataContainerOpenCLFloat>(
            std::vector{ m_inputSize[0], m_inputSize[1], m_inputSize[2] });
        mem->Allocate(m_openclWrapper->m_context);
        m_src.push_back(mem);
    }
    else if (m_src.size() == 1)
    {
        auto mem =
            std::make_shared<DataContainerOpenCLFloat>(std::vector{ m_filterSize[0], m_filterSize[1] });
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
            std::vector{ m_inputSize[0], m_inputSize[1], m_filterSize[1] });
        mem->Allocate(m_openclWrapper->m_context);
        m_dest = mem;
    }
}

void PointwiseConvolutionLayer::FillLayerInputFromFile(const std::filesystem::path &inputPath)
{
    auto fullPath = inputPath / (m_name + "_in.npy");
    m_src[1]->LoadFromFile(fullPath, m_openclWrapper->m_commandQueue);
}

void PointwiseConvolutionLayer::FillLayerConstants(const std::filesystem::path &inputPath)
{
    auto fullPath = inputPath / (m_name + "_weights.npy");
    m_src[0]->LoadFromFile(fullPath, m_openclWrapper->m_commandQueue);
}

PointwiseConvolutionLayer::~PointwiseConvolutionLayer() {}
