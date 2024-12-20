#include "dataContainerFloat.h"
#include "logger.h"
#include "numpy.hpp"

#include <algorithm>
#include <iostream>
#include <numeric>

DataContainerFloat::DataContainerFloat(const std::vector<uint32_t> &dimensions) :
    m_buffer{},
    m_dimensions{ dimensions },
    m_totalElements{ std::accumulate(m_dimensions.begin(), m_dimensions.end(), 1u, std::multiplies<uint32_t>()) }
{
}

DataContainerFloat::DataContainerFloat(uint32_t size) :
    m_buffer{},
    m_dimensions{ size },
    m_totalElements{ std::accumulate(m_dimensions.begin(), m_dimensions.end(), 1u, std::multiplies<uint32_t>()) }
{
}

DataContainerFloat::~DataContainerFloat() {}

const std::vector<float> &DataContainerFloat::GetBuffer() const
{
    return m_buffer;
}

const std::vector<uint32_t> &DataContainerFloat::GetDimensions() const
{
    return m_dimensions;
}

void DataContainerFloat::LoadFromFile(const std::filesystem::path &fullPath)
{
    aoba::LoadArrayFromNumpy(fullPath.string(), m_buffer);

    if (m_buffer.size() != m_totalElements)
    {
        ALOG_GPUML("Invalid size %s", fullPath.filename().c_str());
    }
}

void DataContainerFloat::ResetData()
{
    std::fill(m_buffer.begin(), m_buffer.end(), 0.0f);
}

void DataContainerFloat::DisplyData() const
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
                            m_buffer[i * m_dimensions[1] * m_dimensions[2] * m_dimensions[3] +
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
                        "%5.6f\t", m_buffer[i * m_dimensions[1] * m_dimensions[2] + j * m_dimensions[2] + k]);
                }
                ALOG_GPUML_NO_NEWLINE("\n");
            }
            ALOG_GPUML_NO_NEWLINE("\n");
        }
        for (size_t i = 0; i < s; i++)
        {
            ALOG_GPUML_NO_NEWLINE("%5.6f\t", m_buffer[i]);
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
                ALOG_GPUML_NO_NEWLINE("%5.6f\t", m_buffer[i * m_dimensions[1] + j]);
            }
            ALOG_GPUML_NO_NEWLINE("\n");
        }
        for (int i = 0; i < s; i++)
        {
            ALOG_GPUML_NO_NEWLINE("%5.6f\t", m_buffer[i]);
        }
        ALOG_GPUML_NO_NEWLINE("\n");
    }
    else if (m_dimensions.size() == 1)
    {
        auto w{ std::min(5u, m_totalElements) };
        ALOG_GPUML_NO_NEWLINE("\n");
        for (int i = 0; i < w; i++)
        {
            ALOG_GPUML_NO_NEWLINE("%5.6f\t", m_buffer[i]);
        }
        ALOG_GPUML_NO_NEWLINE("\n");
    }
    else
    {
        std::cerr << "Unknown dimention to print!" << std::endl;
    }
}

void DataContainerFloat::ExportDataInBin(const std::filesystem::path &filePath) const
{
    const auto kTotalSizeInBytes{ m_totalElements * sizeof(float) };
    std::ofstream fileToExport(filePath, std::ios::out | std::ios::binary);
    fileToExport.write((char *)m_buffer.data(), kTotalSizeInBytes);
}

void DataContainerFloat::ExportData(const std::filesystem::path &filePath) const
{
    const auto kTotalSizeInBytes{ m_totalElements * sizeof(float) };
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
                        outfile << m_buffer[i * m_dimensions[1] * m_dimensions[2] * m_dimensions[3] +
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
                    outfile << m_buffer[i * m_dimensions[1] * m_dimensions[2] + j * m_dimensions[2] + k];
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
                outfile << m_buffer[i * m_dimensions[1] + j];
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
            outfile << m_buffer[i];
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

void DataContainerFloat::Allocate()
{
    m_buffer.resize(m_totalElements);
}