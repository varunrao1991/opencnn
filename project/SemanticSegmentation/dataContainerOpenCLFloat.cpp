#include "dataContainerOpenCLFloat.h"
#include "logger.h"
#include "numpy.hpp"

#include <algorithm>
#include <iostream>
#include <memory>
#include <numeric>

DataContainerOpenCLFloat::DataContainerOpenCLFloat(const std::vector<uint32_t> &dimensions) :
    m_buffer{ nullptr },
    m_dimensions{ dimensions },
    m_totalElements{ std::accumulate(m_dimensions.begin(), m_dimensions.end(), 1u, std::multiplies<uint32_t>()) }
{
}

DataContainerOpenCLFloat::DataContainerOpenCLFloat(uint32_t size) :
    m_buffer{ nullptr },
    m_dimensions{ size },
    m_totalElements{ std::accumulate(m_dimensions.begin(), m_dimensions.end(), 1u, std::multiplies<uint32_t>()) }
{
}

DataContainerOpenCLFloat::~DataContainerOpenCLFloat()
{
    if (m_buffer != nullptr)
    {
        auto err = clReleaseMemObject(m_buffer);
        ALOG_GPUML_CHECK_ERROR2(err, CL_SUCCESS);
    }
}

const cl_mem &DataContainerOpenCLFloat::GetBuffer() const
{
    return m_buffer;
}

const std::vector<uint32_t> &DataContainerOpenCLFloat::GetDimensions() const
{
    return m_dimensions;
}

void DataContainerOpenCLFloat::LoadFromFile(const std::filesystem::path &fullPath, cl_command_queue commandQueue)
{
    std::vector<float> iData;
    aoba::LoadArrayFromNumpy(fullPath.string(), iData);

    if (iData.size() == m_totalElements)
    {
        cl_int err = clEnqueueWriteBuffer(
            commandQueue, m_buffer, CL_TRUE, 0, m_totalElements * sizeof(float), iData.data(), 0, nullptr, nullptr);
        ALOG_GPUML_CHECK_ERROR2(err, CL_SUCCESS);
    }
    else
    {
        ALOG_GPUML("Invalid size %s", fullPath.filename().c_str());
    }
}

void DataContainerOpenCLFloat::ResetData(cl_command_queue commandQueue)
{
    const auto kTotalSizeInBytes{ m_totalElements * sizeof(float) };
    auto buffer{ std::make_unique<float[]>(m_totalElements) };
    cl_int err = clEnqueueWriteBuffer(commandQueue, m_buffer, CL_TRUE, 0, kTotalSizeInBytes, buffer.get(), 0, 0, 0);
    ALOG_GPUML_CHECK_ERROR2(err, CL_SUCCESS);
}

void DataContainerOpenCLFloat::FillData(cl_command_queue commandQueue, float *buffer)
{
    const auto kTotalSizeInBytes{ m_totalElements * sizeof(float) };
    cl_int err = clEnqueueWriteBuffer(commandQueue, m_buffer, CL_TRUE, 0, kTotalSizeInBytes, buffer, 0, 0, 0);
    ALOG_GPUML_CHECK_ERROR2(err, CL_SUCCESS);
}

void DataContainerOpenCLFloat::DisplyData(cl_command_queue commandQueue) const
{
    ALOG_GPUML_NO_NEWLINE("Writing buffer of dimention [");
    for (auto index = 0; index < m_dimensions.size(); index++)
    {
        if (index != m_dimensions.size() - 1)
        {
            ALOG_GPUML_NO_NEWLINE("%u, ", m_dimensions[index]);
        }
        else
        {
            ALOG_GPUML("%u]", m_dimensions[index]);
        }
    }

    const auto kTotalSizeInBytes{ m_totalElements * sizeof(float) };
    auto buffer{ std::make_unique<float[]>(m_totalElements) };
    cl_int err = clEnqueueReadBuffer(commandQueue, m_buffer, CL_TRUE, 0, kTotalSizeInBytes, buffer.get(), 0, 0, 0);
    ALOG_GPUML_CHECK_ERROR2(err, CL_SUCCESS);

    if (m_dimensions.size() == 4)
    {
        auto w{ std::min(7u, m_dimensions[0]) };
        auto h{ std::min(7u, m_dimensions[1]) };
        auto m{ std::min(6u, m_dimensions[2]) };
        auto n{ std::min(2u, m_dimensions[3]) };

        for (size_t l = 0; l < n; l++)
        {
            for (size_t k = 0; k < m; k++)
            {
                for (size_t j = 0; j < h; j++)
                {
                    for (size_t i = 0; i < w; i++)
                    {
                        ALOG_GPUML_NO_NEWLINE("%5.6f\t",
                            buffer[i * m_dimensions[1] * m_dimensions[2] * m_dimensions[3] +
                                j * m_dimensions[2] * m_dimensions[3] + k * m_dimensions[3] + l]);
                    }
                    ALOG_GPUML_NO_NEWLINE("\n");
                }
                ALOG_GPUML_NO_NEWLINE("\n");
            }
            ALOG_GPUML("-----------------------");
        }
    }
    else if (m_dimensions.size() == 3)
    {
        auto h{ std::min(4u, m_dimensions[1]) };
        auto w{ std::min(6u, m_dimensions[0]) };
        auto m{ std::min(3u, m_dimensions[2]) };
        auto s{ 10u };

        ALOG_GPUML_NO_NEWLINE("\n");
        for (size_t k = 0; k < m; k++)
        {
            for (size_t j = 0; j < h; j++)
            {
                for (size_t i = 0; i < w; i++)
                {
                    ALOG_GPUML_NO_NEWLINE(
                        "%5.6f\t", buffer[i * m_dimensions[1] * m_dimensions[2] + j * m_dimensions[2] + k]);
                }
                ALOG_GPUML_NO_NEWLINE("\n");
            }
            ALOG_GPUML_NO_NEWLINE("\n");
        }
        for (size_t i = 0; i < s; i++)
        {
            ALOG_GPUML_NO_NEWLINE("%5.6f\t", buffer[i]);
        }
        ALOG_GPUML_NO_NEWLINE("\n");
    }
    else if (m_dimensions.size() == 2)
    {
        auto h{ std::min(4u, m_dimensions[1]) };
        auto w{ std::min(6u, m_dimensions[0]) };
        auto s{ 10ull };

        ALOG_GPUML_NO_NEWLINE("\n");
        for (int j = 0; j < h; j++)
        {
            for (int i = 0; i < w; i++)
            {
                ALOG_GPUML_NO_NEWLINE("%5.6f\t", buffer[i * m_dimensions[1] + j]);
            }
            ALOG_GPUML_NO_NEWLINE("\n");
        }
        for (int i = 0; i < s; i++)
        {
            ALOG_GPUML_NO_NEWLINE("%5.6f\t", buffer[i]);
        }
        ALOG_GPUML_NO_NEWLINE("\n");
    }
    else if (m_dimensions.size() == 1)
    {
        auto w{ std::min(5u, m_totalElements) };
        ALOG_GPUML_NO_NEWLINE("\n");
        for (int i = 0; i < w; i++)
        {
            ALOG_GPUML_NO_NEWLINE("%5.6f\t", buffer[i]);
        }
        ALOG_GPUML_NO_NEWLINE("\n");
    }
    else
    {
        std::cerr << "Unknown dimention to print!" << std::endl;
    }
}

void DataContainerOpenCLFloat::ExportDataInBin(
    const std::filesystem::path &filePath, cl_command_queue commandQueue) const
{
    const auto kTotalSizeInBytes{ m_totalElements * sizeof(float) };
    auto buffer{ std::make_unique<float[]>(m_totalElements) };
    cl_int err = clEnqueueReadBuffer(commandQueue, m_buffer, CL_TRUE, 0, kTotalSizeInBytes, buffer.get(), 0, 0, 0);
    ALOG_GPUML_CHECK_ERROR2(err, CL_SUCCESS);
    std::ofstream fileToExport(filePath, std::ios::out | std::ios::binary);
    fileToExport.write((char *)buffer.get(), kTotalSizeInBytes);
}

void DataContainerOpenCLFloat::ExportData(const std::filesystem::path &filePath, cl_command_queue commandQueue) const
{
    const auto kTotalSizeInBytes{ m_totalElements * sizeof(float) };
    auto buffer{ std::make_unique<float[]>(m_totalElements) };
    cl_int err = clEnqueueReadBuffer(commandQueue, m_buffer, CL_TRUE, 0, kTotalSizeInBytes, buffer.get(), 0, 0, 0);
    ALOG_GPUML_CHECK_ERROR2(err, CL_SUCCESS);

    std::ofstream outfile{ filePath };
    outfile << std::setprecision(4);

    if (m_dimensions.size() == 4)
    {
        auto w{ m_dimensions[0] };
        auto h{ m_dimensions[1] };
        auto m{ m_dimensions[2] };
        auto n{ m_dimensions[3] };

        for (size_t l = 0; l < n; l++)
        {
            for (size_t k = 0; k < m; k++)
            {
                for (size_t j = 0; j < h; j++)
                {
                    for (size_t i = 0; i < w; i++)
                    {
                        outfile << buffer[i * m_dimensions[1] * m_dimensions[2] * m_dimensions[3] +
                            j * m_dimensions[2] * m_dimensions[3] + k * m_dimensions[3] + l];
                        if (i != w - 1)
                        {
                            outfile << ",";
                        }
                    }
                    outfile << std::endl;
                }
            }
        }
    }
    else if (m_dimensions.size() == 3)
    {
        auto h{ m_dimensions[1] };
        auto w{ m_dimensions[0] };
        auto m{ m_dimensions[2] };

        for (int k = 0; k < m; k++)
        {
            for (int j = 0; j < h; j++)
            {
                for (int i = 0; i < w; i++)
                {
                    outfile << buffer[i * m_dimensions[1] * m_dimensions[2] + j * m_dimensions[2] + k];
                    if (i != w - 1)
                    {
                        outfile << ",";
                    }
                }
                outfile << std::endl;
            }
        }
    }
    else if (m_dimensions.size() == 2)
    {
        auto h{ m_dimensions[1] };
        auto w{ m_dimensions[0] };

        for (int j = 0; j < h; j++)
        {
            for (int i = 0; i < w; i++)
            {
                outfile << buffer[i * m_dimensions[1] + j];
                if (i != w - 1)
                {
                    outfile << ",";
                }
            }
            outfile << std::endl;
        }
    }
    else if (m_dimensions.size() == 1)
    {
        auto w{ m_totalElements };
        for (int i = 0; i < w; i++)
        {
            outfile << buffer[i];
            if (i != w - 1)
            {
                outfile << ",";
            }
        }
    }
    else
    {
        std::cerr << "Unknown dimention to export!" << std::endl;
    }
}

void DataContainerOpenCLFloat::Allocate(cl_context context, bool readOnly)
{
    if (m_buffer != nullptr)
    {
        cl_int err = clReleaseMemObject(m_buffer);
        std::cerr << "Error deleting OpenCL buffer: " << err << std::endl;
    }

    cl_int err;

    m_buffer = clCreateBuffer(
        context, readOnly ? CL_MEM_READ_ONLY : CL_MEM_READ_WRITE, m_totalElements * sizeof(float), nullptr, &err);

    if (err != CL_SUCCESS)
    {
        std::cerr << "Error creating OpenCL buffer: " << err << std::endl;
        m_buffer = nullptr;
    }
}