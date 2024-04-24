
#pragma once

#include <stdint.h>
#include <vector>

class IGraph
{
  public:
    virtual ~IGraph() = default;
    virtual bool GetDimension(std::vector<std::vector<uint32_t>> &) = 0;
    virtual bool GetOutputType(std::vector<std::vector<uint32_t>> &, std::vector<uint32_t> &) = 0;
    virtual void Initialize() = 0;
    virtual void Run(float *dataIn) = 0;
    virtual void CopyData(uint8_t *dataOut, int32_t index = 0) = 0;
    virtual void CleanUp() = 0;
};
