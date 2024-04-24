#pragma once

#include "dataContainerBase.h"

#include <CL/cl.h>

#include <filesystem>
#include <vector>

class DataContainerOpenCLFloat : public DataContainerBase
{
  public:
    DataContainerOpenCLFloat(const std::vector<uint32_t> &dimensions);
    DataContainerOpenCLFloat(uint32_t size);

    ~DataContainerOpenCLFloat();

    const cl_mem &GetBuffer() const;
    void LoadFromFile(const std::filesystem::path &file, cl_command_queue commandQueue);
    void DisplyData(cl_command_queue commandQueue) const;
    void ResetData(cl_command_queue commandQueue);
    void FillData(cl_command_queue commandQueue, float *dataIn);
    void ExportData(const std::filesystem::path &file, cl_command_queue commandQueue) const;
    void ExportDataInBin(const std::filesystem::path &file, cl_command_queue commandQueue) const;

    const std::vector<uint32_t> &GetDimensions() const override;

    void Allocate(cl_context context, bool readOnly = false);

  private:
    cl_mem m_buffer;
    std::vector<uint32_t> m_dimensions;
    uint32_t m_totalElements;
};
