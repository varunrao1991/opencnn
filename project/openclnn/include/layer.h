
#pragma once
#include "IGlobalContext.h"
#include "dataContainerOpenCLFloat.h"
#include "logger.h"
#include "openclwrapper.h"

#include "CL/cl.h"

#include <fstream>
#include <memory>
#include <string>
#include <vector>

class Layer
{
  public:
    explicit Layer(const std::string &name, std::shared_ptr<OpenclWrapper> openclWrapper, uint8_t numberOfInputs = 1);
    virtual ~Layer();

    virtual void SetParameters(const std::vector<std::string> &elements) = 0;
    virtual void EnqueueKernel();
    virtual void CreateBuffers(const std::vector<std::shared_ptr<DataContainerOpenCLFloat>> &src) = 0;
    virtual void FillLayerConstants(const std::filesystem::path &inputPath){};
    virtual void FillLayerInputFromFile(const std::filesystem::path &inputPath){};
    virtual void SetKernelArguments() = 0;
    virtual const std::vector<std::string> &GetKernelNames() = 0;

    const std::vector<std::shared_ptr<DataContainerOpenCLFloat>> &GetDestinationBuffers();

    virtual void DisplayOutputBuffer();
    virtual void DisplayInputBuffer();
    virtual void WriteOutputFile(const uint8_t *){};
    virtual void CopyOutputBuffer(uint8_t *outBuffer, int32_t index){};

    const std::string &GetName();
    void SetKernels(std::vector<cl_kernel> &kernel);
    void SetDebugOn(bool debug) { m_debugOn = debug; }
    bool IsDebugOn() const { return m_debugOn; }

    virtual void GetDimension(std::vector<std::vector<uint32_t>> &){};
    virtual void GetOutputTypeSizesInbyte(std::vector<uint32_t> &){};

  protected:
    uint8_t m_expectedNumberOfInputs;
    std::string m_name;
    bool m_debugOn;
    size_t m_globalSize[3];
    size_t m_globalOffset[3];
    uint32_t m_dimension;
    std::vector<std::shared_ptr<DataContainerOpenCLFloat>> m_dest;
    std::vector<std::shared_ptr<DataContainerOpenCLFloat>> m_src;
    std::vector<cl_kernel> m_kernels;
    std::shared_ptr<OpenclWrapper> m_openclWrapper;
};
