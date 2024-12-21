
#include "openclnn/openclnn.h"
#include "graph.h"

#include <string>

extern "C" IGraph *CreateGraph(const std::filesystem::path &kernelsFolderPath,
    const std::filesystem::path &modelPath,
    const std::filesystem::path &outputDirectory)
{
    return new Graph(kernelsFolderPath, modelPath, outputDirectory);
}