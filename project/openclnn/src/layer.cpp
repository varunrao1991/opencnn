#include "layer.h"

#include <cerrno>
#include <cstring>
#include <iostream>
#include <memory>
#include <string>

Layer::Layer(const std::string &name, std::shared_ptr<OpenclWrapper> openclWrapper, uint8_t numberOfInputs) :
    m_name{ name },
    m_debugOn(false),
    m_openclWrapper{ openclWrapper },
    m_expectedNumberOfInputs{ numberOfInputs },
    m_globalOffset{},
    m_globalSize{},
    m_dimension{},
    m_kernels{},
    m_dest{nullptr}
{
}

void Layer::EnqueueKernel()
{
    cl_int err = clEnqueueNDRangeKernel(
        m_openclWrapper->m_commandQueue, m_kernels[0], m_dimension, m_globalOffset, m_globalSize, 0, 0, 0, nullptr);
    ALOG_GPUML_CHECK_ERROR1(err, __LINE__, "clEnqueueNDRangeKernel");
}

const std::shared_ptr<DataContainerOpenCLFloat> &Layer::GetDestinationBuffer()
{
    return m_dest;
}

const std::string &Layer::GetName()
{
    return m_name;
}

void Layer::SetKernels(std::vector<cl_kernel> &kernels)
{
    m_kernels = kernels;
}

void Layer::DisplayInputBuffer()
{
    ALOG_GPUML("Writing input buffer");
    for (const auto kData : m_src)
    {
        kData->DisplyData(m_openclWrapper->m_commandQueue);
    }
}

void Layer::DisplayOutputBuffer()
{
    ALOG_GPUML("Writing output buffer");
    m_dest->DisplyData(m_openclWrapper->m_commandQueue);
}

Layer::~Layer()
{
    ALOG_GPUML("Deleting layer: %s", m_name.c_str());
}
