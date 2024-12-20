#include "pointwiseConvolutionLayerBias.h"
#include "bmpreader.h"
#include "numpy.hpp"

PointwiseConvolutionLayerBias::PointwiseConvolutionLayerBias(
    const std::string &name, std::shared_ptr<OpenclWrapper> openclWrapper) :
    Layer(name, openclWrapper), m_inputSize{}, m_filterSize{}, m_outputSize{}, m_enableRelu{ false }, m_useEven{ false }
{
}

static const std::vector<std::string> s_kKernelNames = { "pointwiseBias", "pointwiseBias1" };
const std::vector<std::string> &PointwiseConvolutionLayerBias::GetKernelNames()
{
    return s_kKernelNames;
}

void PointwiseConvolutionLayerBias::SetParameters(const std::vector<std::string> &elements)
{
    if (elements.size() != 11)
    {
        ALOG_GPUML("PointwiseConvolutionLayerBias element size must be 11 provided %zd", elements.size());
        return;
    }
    if (elements[1] == elements[2] && elements[1] == "1" && elements[elements.size() - 3] == "1" &&
        elements[elements.size() - 2] == "True")
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
        m_outputSize[0] = m_inputSize[0];
        m_outputSize[1] = m_inputSize[1];
        m_outputSize[2] = m_filterSize[1];
        m_useEven = m_filterSize[1] % 2;
    }
    else
    {
        ALOG_GPUML("PointwiseConvolutionLayerBias filter dimension and input dimension mismatch '%s'", m_name.c_str());
    }
}

void PointwiseConvolutionLayerBias::EnqueueKernel()
{
    cl_int err = clEnqueueNDRangeKernel(m_openclWrapper->m_commandQueue,
        m_kernels[m_useEven],
        m_dimension,
        m_globalOffset,
        m_globalSize,
        0,
        0,
        0,
        nullptr);
    ALOG_GPUML_CHECK_ERROR1(err, __LINE__, "clEnqueueNDRangeKernel");
}

void PointwiseConvolutionLayerBias::EnableReluActivation()
{
    m_enableRelu = true;
}

void PointwiseConvolutionLayerBias::SetKernelArguments()
{
    int argCnt = 0;
    m_dimension = 2;
    m_globalSize[0] = static_cast<size_t>(m_inputSize[0]) * m_inputSize[1];

    m_globalSize[1] = m_useEven != 0 ? m_filterSize[1] : m_filterSize[1] / 2;

    if (m_src.size() != 3)
    {
        ALOG_GPUML("PointwiseConvolutionLayerBias : No src memory is created. Failed to set kernel arguments");
        return;
    }
    if (m_dest.size() != 1)
    {
        ALOG_GPUML("No dest memory is created. Failed to set kernel arguments");
        return;
    }
    clSetKernelArg(m_kernels[m_useEven], argCnt++, sizeof(cl_mem), &(m_src[0]->GetBuffer()));
    clSetKernelArg(m_kernels[m_useEven], argCnt++, sizeof(cl_mem), &(m_src[1]->GetBuffer()));
    clSetKernelArg(m_kernels[m_useEven], argCnt++, sizeof(cl_mem), &(m_src[2]->GetBuffer()));
    clSetKernelArg(m_kernels[m_useEven], argCnt++, sizeof(cl_mem), &(m_dest[0]->GetBuffer()));
    clSetKernelArg(m_kernels[m_useEven], argCnt++, sizeof(uint32_t), &m_inputSize[0]);
    clSetKernelArg(m_kernels[m_useEven], argCnt++, sizeof(uint32_t), &m_inputSize[1]);
    clSetKernelArg(m_kernels[m_useEven], argCnt++, sizeof(uint32_t), &m_filterSize[0]);
    clSetKernelArg(m_kernels[m_useEven], argCnt++, sizeof(uint32_t), &m_filterSize[1]);
    clSetKernelArg(m_kernels[m_useEven], argCnt++, sizeof(cl_char), &m_enableRelu);
}

void PointwiseConvolutionLayerBias::CreateBuffers(const std::vector<std::shared_ptr<DataContainerOpenCLFloat>> &src)
{
    m_src = src;
    if (m_src.empty())
    {
        auto mem =
            std::make_shared<DataContainerOpenCLFloat>(std::vector{ m_filterSize[0], m_filterSize[1] });
        mem->Allocate(m_openclWrapper->m_context);
        m_src.push_back(mem);
        mem = std::make_shared<DataContainerOpenCLFloat>(m_filterSize[1]);
        mem->Allocate(m_openclWrapper->m_context);
        m_src.push_back(mem);
        mem = std::make_shared<DataContainerOpenCLFloat>(
            std::vector{ m_inputSize[0], m_inputSize[1], m_inputSize[2] });
        mem->Allocate(m_openclWrapper->m_context);
        m_src.push_back(mem);
    }
    else if (m_src.size() == 1)
    {
        auto mem = std::make_shared<DataContainerOpenCLFloat>(m_filterSize[1]);
        mem->Allocate(m_openclWrapper->m_context);
        m_src.insert(m_src.begin(), mem);
        mem = std::make_shared<DataContainerOpenCLFloat>(std::vector{ m_filterSize[0], m_filterSize[1] });
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
            std::vector{ m_inputSize[0], m_inputSize[1], m_filterSize[1] });
        mem->Allocate(m_openclWrapper->m_context);
        m_dest.push_back(mem);
    }
}

void PointwiseConvolutionLayerBias::FillLayerInputFromFile(const std::filesystem::path &inputPath)
{
    auto fullPath = inputPath / (m_name + "_in.npy");
    m_src[2]->LoadFromFile(fullPath, m_openclWrapper->m_commandQueue);
}

void PointwiseConvolutionLayerBias::FillLayerConstants(const std::filesystem::path &inputPath)
{
    auto fullPath = inputPath / (m_name + "_weights.npy");
    m_src[0]->LoadFromFile(fullPath, m_openclWrapper->m_commandQueue);

    fullPath = inputPath / (m_name + "_bias.npy");
    m_src[1]->LoadFromFile(fullPath, m_openclWrapper->m_commandQueue);
}

PointwiseConvolutionLayerBias::~PointwiseConvolutionLayerBias() {}
