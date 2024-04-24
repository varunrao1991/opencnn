#pragma once

#include "dataContainerBase.h"

#include <filesystem>
#include <vector>

class DataContainerFloat : public DataContainerBase
{
  public:
    DataContainerFloat(const std::vector<uint32_t> &dimensions);
    DataContainerFloat(uint32_t size);

    ~DataContainerFloat();

    const std::vector<float> &GetBuffer() const;
    void LoadFromFile(const std::filesystem::path &file);
    void DisplyData() const;
    void ResetData();
    void ExportData(const std::filesystem::path &file) const;
    void ExportDataInBin(const std::filesystem::path &file) const;

    const std::vector<uint32_t> &GetDimensions() const override;

    void Allocate();

  private:
    std::vector<float> m_buffer;
    std::vector<uint32_t> m_dimensions;
    uint32_t m_totalElements;
};
