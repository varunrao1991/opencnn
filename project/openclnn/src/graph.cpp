#include "graph.h"
#include "argmax.h"
#include "bmpreader.h"
#include "input.h"
#include "layerContext.h"
#include "layerFactory.h"
#include "logger.h"
#include "ssdDecoder.h"

#include "CL/cl.h"

#include <algorithm>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <sstream>
#include <stdlib.h>
#include <string>
#include <thread>
#include <vector>

Graph::Graph(const std::filesystem::path &kernelsFolderPath, const std::filesystem::path &modelPath, const std::filesystem::path &outputDirectory) :
    m_outWidth{ 0 },
    m_outHeight{ 0 },
    m_globalContext{ std::make_shared<GlobalContext>(kernelsFolderPath, modelPath, outputDirectory) },
    m_openclWrapper{ std::make_shared<OpenclWrapper>() },
    m_layerContext{ std::make_shared<LayerContext>(m_globalContext, m_openclWrapper) }
{
    ALOG_GPUML("Model folder : %s", m_modelPath.c_str());
    m_layerContext->Initialize();
}

std::vector<std::string> Graph::splitCommaSeparatedValues(const std::string &input, char delimiter)
{
    std::vector<std::string> values;
    std::stringstream ss(input);
    std::string token;

    while (std::getline(ss, token, delimiter))
    {
        values.push_back(token);
    }
    return values;
}

void Graph::SetKernels()
{
    const auto &kernelMap = m_layerContext->GetKernels();
    for (auto &item : m_layerMap)
    {
        auto kernelNames = item.second->GetKernelNames();
        std::vector<cl_kernel> kernelsOfInterest;
        for (const auto &kernelName : kernelNames)
        {
            auto kernelFound = kernelMap.find(kernelName);
            if (kernelFound == kernelMap.end())
            {
                ALOG_GPUML("Error setting kernel %s", kernelName);
                std::runtime_error("Error setting kernel");
            }
            kernelsOfInterest.push_back((*kernelFound).second);
        }
        item.second->SetKernels(kernelsOfInterest);
    }
}

void Graph::Initialize()
{
    std::filesystem::path graphFileName = m_globalContext->GetGraphFile();
    ALOG_GPUML("Graph file is %s", graphFileName.string().c_str());
    std::ifstream myFile(graphFileName);
    if (myFile.is_open())
    {
        std::string line;
        while (getline(myFile, line))
        {
            if (line[0] == '#')
            {
                continue;
            }
            auto parameters = splitCommaSeparatedValues(line);

            if (parameters.size() > 2)
            {
                size_t beginningIndex = 0;
                bool debugOn = false;
                if (parameters[0] == "?")
                {
                    debugOn = true;
                    beginningIndex = 1;
                }
                const auto kLayerType = parameters[beginningIndex];
                const auto kLayerName = parameters[beginningIndex + 1];
                const auto kLayerInputs = splitCommaSeparatedValues(parameters[beginningIndex + 2], '&');

                std::vector layerParameters(parameters.begin() + beginningIndex + 3, parameters.end());
                auto layer = CreateLayer(kLayerType, kLayerName, layerParameters, m_openclWrapper);

                if (layer != nullptr)
                {
                    layer->SetDebugOn(debugOn);
                    m_layerMap.push_back(std::make_pair(kLayerName, std::move(layer)));
                    m_inputMap.push_back(std::make_pair(kLayerName, kLayerInputs));
                }
                else
                {
                    ALOG_GPUML("Layer is not created : %s", kLayerName.c_str());
                }
            }
        }
        myFile.close();
        SetKernels();
        Create();
        ConnectNodes();
        Reset();
        CreateBuffer(m_linkedLists[0]);
        LoadVariables();
    }
    else
    {
        ALOG_GPUML("Unable to open file %s", graphFileName.string().c_str());
    }
}

void Graph::ConnectNodes()
{
    for (size_t i = 0; i < m_linkedLists.size(); i++)
    {
        auto nodeToConnect = m_linkedLists[i];
        auto inputNodesToFind = m_inputMap[i].second;
        auto layerToSearchInputFor = m_inputMap[i].first;
        for (const auto &kInputLayer : inputNodesToFind)
        {
            if (kInputLayer != "null")
            {
                auto foundLayer = std::find_if(m_inputMap.begin(),
                    m_inputMap.end(),
                    [kInputLayer](auto &node) { return node.first == kInputLayer; });
                if (foundLayer != m_inputMap.end())
                {
                    auto locationIs = foundLayer - m_inputMap.begin();
                    nodeToConnect->AddInput(m_linkedLists[locationIs]);
                }
                else
                {
                    ALOG_GPUML("Fail to find the input %s", kInputLayer.c_str());
                }
            }
        }
    }
}

void Graph::Create()
{
    for (auto &item : m_layerMap)
    {
        m_linkedLists.push_back(std::make_shared<LinkedList<Layer>>(item.second));
    }
}

std::string removeFileExtension(const std::string &fileName)
{
    size_t dotPos = fileName.find_last_of(".");
    if (dotPos != std::string::npos)
    {
        return fileName.substr(0, dotPos);
    }
    else
    {
        return fileName;
    }
}

void Graph::LoadVariables()
{
    auto modelPath = m_globalContext->GetModelPath();
    for (auto &item : m_layerMap)
    {
        item.second->FillLayerConstants(modelPath);
    }
}

void Graph::Print(std::shared_ptr<LinkedList<Layer>> link)
{
    if (!link->IsProcessed())
    {
        auto &inputLinks = link->GetInputs();
        std::vector<cl_mem> accumulatedDestinationBuffers;
        for (auto inputLink : inputLinks)
        {
            if (!inputLink->IsProcessed())
            {
                Print(inputLink);
            }
        }
        ALOG_GPUML("%s", link->GetEntity()->GetName().c_str());
        link->SetProcessed();
    }
    auto &outputLinks = link->GetOutputs();
    for (auto outputLink : outputLinks)
    {
        Print(outputLink);
    }
}

void Graph::CreateBuffer(std::shared_ptr<LinkedList<Layer>> link)
{
    if (!link->IsProcessed())
    {
        auto &inputLinks = link->GetInputs();
        std::vector<std::shared_ptr<DataContainerOpenCLFloat>> accumulatedDestinationBuffers;
        for (auto inputLink : inputLinks)
        {
            if (!inputLink->IsProcessed())
            {
                CreateBuffer(inputLink);
            }
            auto destinationBuffers = inputLink->GetEntity()->GetDestinationBuffers();
            for (auto &destinationBuffer : destinationBuffers)
            {
                accumulatedDestinationBuffers.push_back(destinationBuffer);
            }
        }
        //ALOG_GPUML("Creating the buffer for %s", link->GetEntity()->GetName().c_str());
        link->GetEntity()->CreateBuffers(accumulatedDestinationBuffers);
        link->SetProcessed();
    }
    auto &outputLinks = link->GetOutputs();
    for (auto outputLink : outputLinks)
    {
        CreateBuffer(outputLink);
    }
}

void Graph::Reset()
{
    for (auto &item : m_linkedLists)
    {
        item->ResetProcessed();
    }
}

void Graph::Execute(std::shared_ptr<LinkedList<Layer>> link)
{
    auto &inputLinks = link->GetInputs();
    for (auto inputLink : inputLinks)
    {
        if (!inputLink->IsProcessed())
        {
            Execute(inputLink);
        }
    }

    if (!link->IsProcessed())
    {
        link->GetEntity()->SetKernelArguments();
        link->GetEntity()->EnqueueKernel();
        if (link->GetEntity()->IsDebugOn())
        {
            link->GetEntity()->DisplayInputBuffer();
            link->GetEntity()->DisplayOutputBuffer();
        }
        link->SetProcessed();
    }
    auto &outputLinks = link->GetOutputs();
    for (auto outputLink : outputLinks)
    {
        Execute(outputLink);
    }
}

void Graph::CopyData(uint8_t *dataOut, int32_t index)
{
    m_layerMap[m_layerMap.size() - 1].second->CopyOutputBuffer(dataOut, index);
}

void Graph::Run(float *dataIn)
{
    auto layer = dynamic_cast<InputLayer *>(m_layerMap[0].second.get());
    if (layer != nullptr)
    {
        layer->FillInputFromBuffer(dataIn);
    }
    else
    {
        ALOG_GPUML("1st layer must be input layer");
    }
    Reset();
    Execute(m_linkedLists[0]);
}

bool Graph::GetDimension(std::vector<std::vector<uint32_t>> &sizeIn)
{
    if (!m_layerMap.empty())
    {
        m_layerMap[0].second->GetDimension(sizeIn);
        return true;
    }
    else
    {
        return false;
    }
}

bool Graph::GetOutputType(std::vector<std::vector<uint32_t>> &dimentionOut, std::vector<uint32_t> &sizeOut)
{
    if (!m_layerMap.empty())
    {
        m_layerMap[m_layerMap.size() - 1].second->GetDimension(dimentionOut);
        m_layerMap[m_layerMap.size() - 1].second->GetOutputTypeSizesInbyte(sizeOut);
        return true;
    }
    else
    {
        return false;
    }
}

void Graph::CleanUp()
{
    m_layerMap.clear();
}

Graph::~Graph()
{
    m_layerContext->CleanUp();
    ALOG_GPUML("Graph deleted : %s", m_graphFileName.c_str());
}
