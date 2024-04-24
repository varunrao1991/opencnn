
#include "SemanticSegmentationAPI.h"
#include "graph.h"

#include <string>

extern "C" IGraph *CreateGraph(const std::string &graphFile, const std::string &basePath)
{
    return new Graph(graphFile, basePath);
}