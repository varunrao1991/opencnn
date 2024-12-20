#pragma once

#include "globalContext.h"
#include "logger.h"
#include "openclwrapper.h"

#include "CL/cl.h"

#include <fstream>
#include <map>
#include <memory>
#include <string>
#include <vector>

class ILayerContext
{
  public:
    virtual ~ILayerContext() = default;
    virtual void Initialize() = 0;
    virtual void CleanUp() = 0;
    virtual const std::map<const std::string, cl_kernel> &GetKernels() = 0;
};
