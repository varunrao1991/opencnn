#pragma once

#include "IGlobalContext.h"
#include "ILayerContext.h"
#include "logger.h"
#include "openclwrapper.h"

#include "CL/cl.h"

#include <fstream>
#include <map>
#include <memory>
#include <string>
#include <vector>

class LayerContext : public ILayerContext
{
  public:
    explicit LayerContext(std::shared_ptr<IGlobalContext> globalContext, std::shared_ptr<OpenclWrapper> openclWrapper);
    void Initialize() override;
    void CleanUp() override;
    const std::map<const std::string, cl_kernel> &GetKernels();
    virtual ~LayerContext();

  protected:
    void BuildProgramFromFile(const std::filesystem::path &kernelFilePath);
    void BuildKernels();

    std::shared_ptr<IGlobalContext> m_globalContext;
    std::shared_ptr<OpenclWrapper> m_openclWrapper;

  private:
    bool loadProgSource(const std::filesystem::path &filePath, cl_program &program);
    std::map<const std::string, cl_program> m_programMaps;
    std::map<const std::string, cl_kernel> m_kernelMaps;
};
