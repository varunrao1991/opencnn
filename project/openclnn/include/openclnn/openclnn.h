#pragma once

#include <filesystem>

class IGraph;

#ifdef MYDLL_EXPORTS
#define MYDLL_API __declspec(dllexport)
#else
#define MYDLL_API __declspec(dllimport)
#endif

extern "C" MYDLL_API IGraph *CreateGraph(const std::filesystem::path &kernelsFolderPath,
    const std::filesystem::path &modelPath,
    const std::filesystem::path &outputDirectory);