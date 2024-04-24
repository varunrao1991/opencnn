#include "layerContext.h"
#include "openclwrapper.h"

#include <filesystem>
#include <iostream>
#include <memory>
#include <string>

LayerContext::LayerContext(
    std::shared_ptr<IGlobalContext> globalContext, std::shared_ptr<OpenclWrapper> openclWrapper) :
    m_globalContext(globalContext), m_openclWrapper(openclWrapper), m_programMaps{}, m_kernelMaps{}
{
}

void LayerContext::Initialize()
{
    constexpr auto kFileExtension{ ".cl" };
    const auto kKernelDirectory{ m_globalContext->GetKernelPath() };
    if (std::filesystem::exists(kKernelDirectory))
    {
        ALOG_GPUML("Loading kernels from directory : %s", std::filesystem::absolute(kKernelDirectory).string().c_str());
        for (const auto &entry : std::filesystem::directory_iterator(kKernelDirectory))
        {
            if (entry.is_regular_file() && entry.path().extension() == kFileExtension)
            {
                BuildProgramFromFile(entry.path());
            }
        }
    }
    else
    {
        ALOG_GPUML("Kernel directory is invalid : %s", kKernelDirectory.c_str());
        std::runtime_error("Kernel directory is invalid");
    }
    BuildKernels();
}

bool LayerContext::loadProgSource(const std::filesystem::path &filePath, cl_program &program)
{
    size_t szFinalLength;
    FILE *pFileStream = nullptr;
    char cPreamble[] = "";
    size_t szSourceLength;

    auto error_s = fopen_s(&pFileStream, filePath.string().c_str(), "rb");
    if (error_s != 0)
    {
        return false;
    }

    size_t szPreambleLength = std::strlen(cPreamble);

    fseek(pFileStream, 0, SEEK_END);
    szSourceLength = ftell(pFileStream);
    fseek(pFileStream, 0, SEEK_SET);

    char *cSourceString = (char *)malloc(szSourceLength + szPreambleLength + 1);
    if (cSourceString == nullptr)
    {
        return false;
    }
    memcpy(cSourceString, cPreamble, szPreambleLength);
    if (fread((cSourceString) + szPreambleLength, szSourceLength, 1, pFileStream) != 1)
    {
        fclose(pFileStream);
        free(cSourceString);
        return false;
    }

    fclose(pFileStream);
    szFinalLength = szSourceLength + szPreambleLength;

    cSourceString[szSourceLength + szPreambleLength] = '\0';

    cl_int err;
    program = clCreateProgramWithSource(m_openclWrapper->m_context, 1, (const char **)&cSourceString, NULL, &err);
    ALOG_GPUML_CHECK_ERROR1(err, __LINE__, "clCreateProgramWithSource");
    free(cSourceString);
    return CL_SUCCESS == err;
}

void LayerContext::BuildProgramFromFile(const std::filesystem::path &kernelFilePath)
{
    cl_int err;
    const auto kKernelName = kernelFilePath.filename().stem().string();

    std::filesystem::path binaryPath = std::filesystem::path(kernelFilePath).replace_extension(".bin");

    cl_program program;
    if (std::filesystem::exists(binaryPath))
    {
        std::ifstream file(binaryPath, std::ios::binary);

        if (!file)
        {
            std::cerr << "Failed to open file." << std::endl;
            return;
        }

        file.seekg(0, std::ios::end);
        std::streampos fileSize = file.tellg();
        file.seekg(0, std::ios::beg);

        auto buffer = (char *)malloc(fileSize);
        file.read(buffer, fileSize);
        cl_int status;
        auto bufferSize = static_cast<size_t>(fileSize);
        program = clCreateProgramWithBinary(m_openclWrapper->m_context,
            1,
            &m_openclWrapper->m_deviceId,
            &bufferSize,
            (const unsigned char **)&buffer,
            &status,
            &err);

        free(buffer);
        ALOG_GPUML_CHECK_ERROR1(err, __LINE__, "clCreateProgramWithBinary");
        m_programMaps.insert(std::make_pair(kKernelName, program));
        err = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
        ALOG_GPUML_CHECK_ERROR1(err, __LINE__, "clBuildProgram");
    }
    else
    {
        if (!loadProgSource(kernelFilePath, program))
        {
            ALOG_GPUML("Missing source file %s?", kernelFilePath.string().c_str());
            return;
        }
        m_programMaps.insert(std::make_pair(kKernelName, program));

        err = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);

        cl_build_status build_status;
        err = clGetProgramBuildInfo(program,
            m_openclWrapper->m_deviceId,
            CL_PROGRAM_BUILD_STATUS,
            sizeof(cl_build_status),
            &build_status,
            NULL);
        ALOG_GPUML_CHECK_ERROR1(err, __LINE__, "clGetProgramBuildInfo");

        size_t ret_val_size;
        err = clGetProgramBuildInfo(program, m_openclWrapper->m_deviceId, CL_PROGRAM_BUILD_LOG, 0, NULL, &ret_val_size);
        ALOG_GPUML_CHECK_ERROR1(err, __LINE__, "clGetProgramBuildInfo");
        auto build_log = std::make_unique<char[]>(ret_val_size + 1);
        err = clGetProgramBuildInfo(
            program, m_openclWrapper->m_deviceId, CL_PROGRAM_BUILD_LOG, ret_val_size, build_log.get(), NULL);
        ALOG_GPUML_CHECK_ERROR1(err, __LINE__, "clGetProgramBuildInfo");

        build_log[ret_val_size - 1] = '\0';

        ALOG_GPUML("%s", build_log.get());

        size_t binarySize;
        err = clGetProgramInfo(program, CL_PROGRAM_BINARY_SIZES, sizeof(size_t), &binarySize, NULL);
        ALOG_GPUML_CHECK_ERROR1(err, __LINE__, "clGetProgramInfo");

#if 1
        if (binarySize > 0)
        {
            unsigned char *programBinary = (unsigned char *)malloc(binarySize);
            err = clGetProgramInfo(program, CL_PROGRAM_BINARIES, sizeof(unsigned char *), &programBinary, NULL);
            ALOG_GPUML_CHECK_ERROR1(err, __LINE__, "clGetProgramInfo");
            if (err == CL_SUCCESS)
            {
                FILE *pFile;
                auto status_s = fopen_s(&pFile, binaryPath.string().c_str(), "wb");
                if (pFile)
                {
                    fwrite(programBinary, binarySize, sizeof(unsigned char), pFile);
                    fclose(pFile);
                    ALOG_GPUML("File written to folder %s", binaryPath.string().c_str());
                }
            }
            free(programBinary);
        }
#endif
    }
    return;
}

void LayerContext::BuildKernels()
{
    for (const auto &programPair : m_programMaps)
    {
        auto program = programPair.second;
        auto kernelName = programPair.first;

        cl_int err = CL_SUCCESS;
        cl_kernel kernel = clCreateKernel(program, kernelName.c_str(), &err);
        ALOG_GPUML_CHECK_ERROR1(err, __LINE__, "clCreateKernel");
        if (err == CL_SUCCESS)
        {
            m_kernelMaps.insert(std::make_pair(kernelName, kernel));
        }
    }
}

const std::map<const std::string, cl_kernel> &LayerContext::GetKernels()
{
    return m_kernelMaps;
}

void LayerContext::CleanUp()
{
    for (const auto &programPair : m_programMaps)
    {
        cl_int clStatus = clReleaseProgram(programPair.second);
        ALOG_GPUML_CHECK_ERROR1(clStatus, __LINE__, "Program Released");
    }
    for (const auto [kernelName, kernel] : m_kernelMaps)
    {
        cl_int clStatus = clReleaseKernel(kernel);
        ALOG_GPUML_CHECK_ERROR1(clStatus, __LINE__, "Kernel Released");
    }
    m_kernelMaps.clear();
    ALOG_GPUML("Layer context cleaned");
}

LayerContext::~LayerContext()
{
    ALOG_GPUML("Layer context deleted : %s", m_globalContext->GetKernelPath().string().c_str());
}
