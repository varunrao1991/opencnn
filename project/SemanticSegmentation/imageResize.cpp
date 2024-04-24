#include "imageResize.h"
#include <iomanip>
#include <numeric>

ImageResize::ImageResize(const std::string &name, std::shared_ptr<OpenclWrapper> openclWrapper) :
    Layer(name, openclWrapper), m_inputSize{ 0, 0, 0 }, m_outputSize{ 0, 0, 0 }
{
}

static const std::vector<std::string> s_kKernelNames = { "resize_normal" };
const std::vector<std::string> &ImageResize::GetKernelNames()
{
    return s_kKernelNames;
}

void ImageResize::SetParameters(const std::vector<std::string> &elements)
{
    if (elements.size() != 6 && elements[2] != elements[5])
    {
        ALOG_GPUML("ImageResize element size must be 6 or mismatch");
        for (int i = 0; i < elements.size(); i++)
        {
            ALOG_GPUML(elements[i].c_str());
        }
        return;
    }

    auto kTotalSizeElements = 6;
    std::vector<int> parameters(kTotalSizeElements);
    int numCount = 0;
    for (int i = 0; i < kTotalSizeElements; i++)
    {
        parameters[numCount++] = std::atoi(elements[i].c_str());
    }
    std::copy(parameters.begin(), parameters.begin() + 3, m_inputSize.begin());
    std::copy(parameters.begin() + 3, parameters.end(), m_outputSize.begin());
}

void ImageResize::SetKernelArguments()
{
    int argCnt = 0;
    m_dimension = 3;
    m_globalSize[0] = m_outputSize[2];
    m_globalSize[1] = m_outputSize[1];
    m_globalSize[2] = m_outputSize[0];
    if (m_src.size() != 1)
    {
        ALOG_GPUML("ImageResize : No src memory is created. Failed to set kernel arguments");
        return;
    }
    if (m_dest.size() != 1)
    {
        ALOG_GPUML("No dest memory is created. Failed to set kernel arguments");
        return;
    }
    clSetKernelArg(m_kernels[0], argCnt++, sizeof(cl_mem), &(m_src[0]->GetBuffer()));
    clSetKernelArg(m_kernels[0], argCnt++, sizeof(cl_mem), &(m_dest[0]->GetBuffer()));
    clSetKernelArg(m_kernels[0], argCnt++, sizeof(uint32_t), &m_inputSize[2]);
    clSetKernelArg(m_kernels[0], argCnt++, sizeof(uint32_t), &m_inputSize[1]);
    clSetKernelArg(m_kernels[0], argCnt++, sizeof(uint32_t), &m_inputSize[0]);
    clSetKernelArg(m_kernels[0], argCnt++, sizeof(uint32_t), &m_outputSize[1]);
    clSetKernelArg(m_kernels[0], argCnt++, sizeof(uint32_t), &m_outputSize[0]);
}
void ImageResize::CreateBuffers(const std::vector<std::shared_ptr<DataContainerOpenCLFloat>> &src)
{
    auto inElements = std::accumulate(m_inputSize.begin(), m_inputSize.end(), 1u, std::multiplies<uint32_t>());
    auto outElements = std::accumulate(m_outputSize.begin(), m_outputSize.end(), 1u, std::multiplies<uint32_t>());
    m_src = src;
    if (m_src.empty())
    {
        auto mem = std::make_shared<DataContainerOpenCLFloat>(inElements);
        mem->Allocate(m_openclWrapper->m_context);
        m_src.push_back(mem);
    }
    if (m_dest.empty())
    {
        auto mem = std::make_shared<DataContainerOpenCLFloat>(outElements);
        mem->Allocate(m_openclWrapper->m_context);
        m_dest.push_back(mem);
    }
}

ImageResize::~ImageResize() {}
