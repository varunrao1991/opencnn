
#pragma once
#include "IGlobalContext.h"
#include "openclnn/IGraph.h"
#include "Layer.h"
#include "LayerContext.h"
#include "LinkedList.h"
#include "logger.h"
#include "openclWrapper.h"

#include <memory>
#include <string>
#include <vector>

class Graph : public IGraph
{
  public:
    Graph(const std::filesystem::path &kernelsFolderPath, const std::filesystem::path &modelPath, const std::filesystem::path &outputDirectory);
    virtual ~Graph();
    bool GetDimension(std::vector<std::vector<uint32_t>> &) override;
    bool GetOutputType(std::vector<std::vector<uint32_t>> &dimentionOut, std::vector<uint32_t> &sizeOut) override;
    void CleanUp() override;
    void Run(float *dataIn) override;
    void CopyData(uint8_t *dataOut, int32_t index = 0) override;
    void Initialize() override;

  private:
    std::vector<std::string> splitCommaSeparatedValues(const std::string &input, char delimiter = ',');

    void Create();
    void ConnectNodes();
    void LoadVariables();
    void SetKernels();
    void Reset();
    void Print(std::shared_ptr<LinkedList<Layer>> link);
    void CreateBuffer(std::shared_ptr<LinkedList<Layer>> link);
    void Execute(std::shared_ptr<LinkedList<Layer>> node);

    std::shared_ptr<IGlobalContext> m_globalContext;
    std::shared_ptr<OpenclWrapper> m_openclWrapper;
    std::shared_ptr<LayerContext> m_layerContext;
    uint32_t m_outWidth;
    uint32_t m_outHeight;
    std::vector<std::pair<const std::string, std::shared_ptr<Layer>>> m_layerMap;
    std::vector<std::pair<const std::string, std::vector<std::string>>> m_inputMap;
    std::vector<std::shared_ptr<LinkedList<Layer>>> m_linkedLists;
};
