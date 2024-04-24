#pragma once

#include <string>

class IGraph;

#ifdef MYDLL_EXPORTS
#define MYDLL_API __declspec(dllexport)
#else
#define MYDLL_API __declspec(dllimport)
#endif

extern "C" MYDLL_API IGraph *CreateGraph(const std::string &graphFile, const std::string &basePath);