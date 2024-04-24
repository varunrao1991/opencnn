#pragma once

#include <cstdint>
#include <vector>

class DataContainerBase
{
  public:
    virtual ~DataContainerBase() = default;

    virtual const std::vector<uint32_t> &GetDimensions() const = 0;
};