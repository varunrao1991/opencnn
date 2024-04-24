#include "layerFactory.h"
#include "add.h"
#include "argmax.h"
#include "batchNormalization.h"
#include "batchNormalization1.h"
#include "batchNormalization2.h"
#include "concatenation.h"
#include "convolutionLayer.h"
#include "convolutionLayerBias.h"
#include "deconvolutionLayer.h"
#include "deconvolutionLayer3X3.h"
#include "depthwiseConvolutionBiasLayer.h"
#include "depthwiseConvolutionLayer.h"
#include "imageResize.h"
#include "input.h"
#include "leakyrelu.h"
#include "maxpool.h"
#include "padding.h"
#include "pointwiseConvolutionLayer.h"
#include "pointwiseConvolutionLayerBias.h"
#include "relu.h"
#include "sigmoidLayer.h"
#include "softmax.h"
#include "ssdDecoder.h"
#include "yoloDecoder.h"
#include "yoloSeperateDecoder.h"

std::unique_ptr<Layer> CreateLayer(const std::string &layerType,
    const std::string &layerName,
    std::vector<std::string> &layerParameters,
    std::shared_ptr<OpenclWrapper> openclWrapper)
{
    std::unique_ptr<Layer> layer = nullptr;
    if (layerType == "InputLayer")
    {
        layer = std::make_unique<InputLayer>(layerName, openclWrapper);
        layer->SetParameters(layerParameters);
    }
    else if (layerType == "Conv2D")
    {
        if (layerParameters.size() < 11)
        {
            ALOG_GPUML("Invalid parameter size");
        }
        else if (layerParameters[1] == layerParameters[2] && layerParameters[1] != "1")
        {
            if (layerParameters[9] == "False")
            {
                layer = std::make_unique<ConvolutionLayer>(layerName, openclWrapper);
                layer->SetParameters(layerParameters);
            }
            else
            {
                layer = std::make_unique<ConvolutionLayerBias>(layerName, openclWrapper);
                layer->SetParameters(layerParameters);
            }
        }
        else if (layerParameters[1] == layerParameters[2] && layerParameters[1] == "1")
        {
            if (layerParameters[9] == "False")
            {
                layer = std::make_unique<PointwiseConvolutionLayer>(layerName, openclWrapper);
                layer->SetParameters(layerParameters);
            }
            else
            {
                layer = std::make_unique<PointwiseConvolutionLayerBias>(layerName, openclWrapper);
                layer->SetParameters(layerParameters);
            }
        }
        else
        {
            ALOG_GPUML("Invalid convolution type, no layer created");
        }
    }
    else if (layerType == "BatchNormalization")
    {
        if (layerParameters[layerParameters.size() - 1] == "2")
        {
            layer = std::make_unique<BatchNormalization2>(layerName, openclWrapper);
            layer->SetParameters(layerParameters);
        }
        else if (layerParameters[layerParameters.size() - 1] == "3")
        {
            layer = std::make_unique<BatchNormalization1>(layerName, openclWrapper);
            layer->SetParameters(layerParameters);
        }
        else
        {
            layer = std::make_unique<BatchNormalization>(layerName, openclWrapper);
            layer->SetParameters(layerParameters);
        }
    }
    else if (layerType == "MaxPooling2D")
    {
        layer = std::make_unique<MaxPool>(layerName, openclWrapper);
        layer->SetParameters(layerParameters);
    }
    else if (layerType == "ReLU")
    {
        layer = std::make_unique<Relu>(layerName, openclWrapper);
        layer->SetParameters(layerParameters);
    }
    else if (layerType == "LeakyReLU")
    {
        layer = std::make_unique<LeakyRelu>(layerName, openclWrapper);
        layer->SetParameters(layerParameters);
    }
    else if (layerType == "ImageResize")
    {
        layer = std::make_unique<ImageResize>(layerName, openclWrapper);
        layer->SetParameters(layerParameters);
    }
    else if (layerType == "DepthwiseConv2D")
    {
        if (layerParameters[layerParameters.size() - 1] == "False")
        {
            layer = std::make_unique<DepthwiseConvolutionLayer>(layerName, openclWrapper);
            layer->SetParameters(layerParameters);
        }
        else
        {
            layer = std::make_unique<DepthwiseConvolutionBiasLayer>(layerName, openclWrapper);
            layer->SetParameters(layerParameters);
        }
    }
    else if (layerType == "ZeroPadding2D")
    {
        layer = std::make_unique<PaddingLayer>(layerName, openclWrapper);
        layer->SetParameters(layerParameters);
    }
    else if (layerType == "Add")
    {
        layer = std::make_unique<Add>(layerName, openclWrapper);
        layer->SetParameters(layerParameters);
    }
    else if (layerType == "Conv2DTranspose")
    {
        if (layerParameters[0] == "3" && layerParameters[1] == "3")
        {
            layer = std::make_unique<DeconvolutionLayer3X3>(layerName, openclWrapper);
            layer->SetParameters(layerParameters);
        }
        else if (layerParameters[0] == "2" && layerParameters[1] == "2")
        {
            layer = std::make_unique<DeconvolutionLayer>(layerName, openclWrapper);
            layer->SetParameters(layerParameters);
        }
    }
    else if (layerType == "Concatenate")
    {
        layer = std::make_unique<Concatenation>(layerName, openclWrapper);
        layer->SetParameters(layerParameters);
    }
    else if (layerType == "SSDDecoder")
    {
        layer = std::make_unique<SSDDecoder>(layerName, openclWrapper);
        layer->SetParameters(layerParameters);
    }
    else if (layerType == "YOLODecoder")
    {
        layer = std::make_unique<YOLODecoder>(layerName, openclWrapper);
        layer->SetParameters(layerParameters);
    }
    else if (layerType == "YOLOSeperateDecoder")
    {
        layer = std::make_unique<YOLOSeperateDecoder>(layerName, openclWrapper);
        layer->SetParameters(layerParameters);
    }
    else if (layerType == "Activation")
    {
        auto newLayerParameters = std::vector<std::string>{ layerParameters.begin() + 1, layerParameters.end() };
        if (layerParameters[0] == "relu")
        {
            layer = std::make_unique<Relu>(layerName, openclWrapper);
            layer->SetParameters(newLayerParameters);
        }
        else if (layerParameters[0] == "sigmoid")
        {
            layer = std::make_unique<SigmoidLayer>(layerName, openclWrapper);
            layer->SetParameters(newLayerParameters);
        }
        else if (layerParameters[0] == "softmax")
        {
            layer = std::make_unique<Softmax>(layerName, openclWrapper);
            layer->SetParameters(newLayerParameters);
        }
        else
        {
            ALOG_GPUML("Invalid activation type, no layer created");
        }
    }
    else if (layerType == "Argmax")
    {
        layer = std::make_unique<ArgMax>(layerName, openclWrapper);
        layer->SetParameters(layerParameters);
    }
    return layer;
}