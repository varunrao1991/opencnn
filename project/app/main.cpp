#include "bmpreader.h"
#include "openclnn/IGraph.h"
#include "openclnn/openclnn.h"

#include <array>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <numeric>
#include <string>

constexpr float kOffset = 0.0f;
constexpr float kDivFactor = 255.0f;

void fillData(std::string fullPath, float *dst, const int dstWidth, const int dstHeight, const uint32_t dstChannels)
{
    BmpFile bmpReader(fullPath);

    int srcWidth;
    int srcHeight;
    auto src = bmpReader.read_bitmap_file_unchar(srcWidth, srcHeight);

    const uint32_t kSrcChannels = 3;
    int xOffset = std::max(0, (dstWidth - srcWidth) / 2);
    int yOffset = std::max(0, (dstHeight - srcHeight) / 2);

    for (int y = 0; y < srcHeight; ++y)
    {
        for (int x = 0; x < srcWidth; ++x)
        {
            int srcIndex = kSrcChannels * (y * srcWidth + x);
            int dstIndex = dstChannels * ((y + yOffset) * dstWidth + (x + xOffset));
            for (size_t ch = 0; ch < std::min(dstChannels, kSrcChannels); ch++)
            {
                dst[dstIndex + ch] = src[srcIndex + ch] / kDivFactor - kOffset;
            }
        }
    }
}

void fillData(float *bufferIn,
    uint8_t *bufferOut,
    uint32_t width,
    uint32_t height,
    const uint32_t channelsIn,
    const uint32_t channelsOut)
{
    for (uint32_t y = 0; y < height; ++y)
    {
        for (uint32_t x = 0; x < width; ++x)
        {
            auto kInIndex = channelsIn * (y * width + x);
            auto kOutIndex = channelsOut * (y * width + x);
            for (size_t ch = 0; ch < std::min(channelsIn, channelsOut); ch++)
            {
                bufferOut[kOutIndex + ch] = static_cast<uint8_t>(kDivFactor * (kOffset + bufferIn[kInIndex + ch]));
            }
        }
    }
}

int main(int argc, char *argv[])
{
    if (argc != 4)
    {
        std::cerr << "Usage: " << argv[0] << " <kernels> <model> <inputBmp>\n";
        return 1;
    }

    const std::filesystem::path kKernelsPath{ argv[1] };
    const std::filesystem::path kModelPath{ argv[2] };
    const std::filesystem::path kInputBmp{ argv[3] };

    std::filesystem::path kOutputPath{ "./tmp" };
    if (!std::filesystem::exists(kOutputPath))
    {
        std::filesystem::create_directory(kOutputPath);
    }

    if (std::filesystem::exists(kKernelsPath) && std::filesystem::exists(kModelPath))
    {
        auto graph = CreateGraph(kKernelsPath, kModelPath, kOutputPath);

        graph->Initialize();

        std::vector<std::vector<uint32_t>> inputDimensions;
        std::vector<std::vector<uint32_t>> outputDimensions;
        std::vector<uint32_t> outputSizes;
        graph->GetDimension(inputDimensions);
        graph->GetOutputType(outputDimensions, outputSizes);
        auto inputDimension = inputDimensions[0];
        auto outputDimension1 = outputDimensions[0];

        auto inputElementsCount =
            std::accumulate(inputDimension.begin(), inputDimension.end(), 1, std::multiplies<int>());
        auto outputElementsCount =
            std::accumulate(outputDimension1.begin(), outputDimension1.end(), 1, std::multiplies<int>());

        std::vector<float> dataInput = std::vector<float>(inputElementsCount);
        std::vector<uint8_t> dataOutput = std::vector<uint8_t>(outputElementsCount);

        fillData(kInputBmp.string(), dataInput.data(), inputDimension[0], inputDimension[1], inputDimension[2]);
        fillData(dataInput.data(),
            dataOutput.data(),
            outputDimension1[0],
            outputDimension1[1],
            inputDimension[2],
            outputDimension1[2]);

        auto start = std::chrono::high_resolution_clock::now();
        graph->Run(dataInput.data());
        auto end = std::chrono::high_resolution_clock::now();
        graph->CopyData(dataOutput.data());

        if (outputDimensions.size() == 3)
        {
            auto boxesCountDimention = outputDimensions[1];
            auto boxesDataDimention = outputDimensions[2];
            auto boxesCountBytesize = outputSizes[1];
            auto boxesDataByteSize = outputSizes[2];
            auto totalSizeForBoxesCount =
                std::accumulate(boxesCountDimention.begin(), boxesCountDimention.end(), 1, std::multiplies<int>());
            auto totalSizeForBoxesData =
                std::accumulate(boxesDataDimention.begin(), boxesDataDimention.end(), 1, std::multiplies<int>());
            auto totalBoxesDetected = std::make_unique<uint8_t[]>(totalSizeForBoxesCount * boxesCountBytesize);
            auto actualBoxesData = std::make_unique<uint8_t[]>(totalSizeForBoxesData * boxesDataByteSize);

            if (boxesCountBytesize == sizeof(uint32_t) && totalSizeForBoxesCount == 1)
            {
                uint32_t *newPtr = (uint32_t *)(totalBoxesDetected.get());
                const uint32_t kBoxParameters = 7;
                graph->CopyData(totalBoxesDetected.get(), 1);
                auto &detectedCount = newPtr[0];

                if (boxesDataByteSize == sizeof(float) && boxesDataDimention[0] == kBoxParameters && detectedCount > 0)
                {
                    graph->CopyData(actualBoxesData.get(), 2);
                    float *newDataPtr = (float *)(actualBoxesData.get());

                    for (size_t boxIndex = 0; boxIndex < detectedCount; boxIndex++)
                    {
                        std::cout << "X1: " << newDataPtr[boxIndex * kBoxParameters + 0]
                                  << ", X2: " << newDataPtr[boxIndex * kBoxParameters + 1]
                                  << ", Y1: " << newDataPtr[boxIndex * kBoxParameters + 2]
                                  << ", Y2: " << newDataPtr[boxIndex * kBoxParameters + 3]
                                  << ", XC: " << newDataPtr[boxIndex * kBoxParameters + 4]
                                  << ", YC: " << newDataPtr[boxIndex * kBoxParameters + 5]
                                  << ", La: " << newDataPtr[boxIndex * kBoxParameters + 6] << std::endl;
                    }
                }
            }
        }

        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        std::cout << "Elapsed time: " << duration.count() << " milliseconds" << std::endl;

        if (true)
        {
            std::filesystem::path filename = kOutputPath / "input.bmp";
            BmpFile bmp(filename.string());
            bmp.write_bitmap(dataInput.data(),
                inputDimension[0],
                inputDimension[1],
                inputDimension[2],
                0,
                1,
                2,
                0.5f,
                0.5f,
                0.5f,
                0.5f,
                0.5f,
                0.5f,
                255);
        }
        if (true)
        {
            std::filesystem::path filename = kOutputPath / "output.bmp";
            BmpFile bmp(filename.string());
            if (outputDimension1[2] == 3)
            {
                bmp.write_bitmap_test(dataOutput.data(), outputDimension1[0], outputDimension1[1]);
            }
            else
            {
                bmp.write_bitmap1ch(dataOutput.data(), outputDimension1[0], outputDimension1[1], 1);
            }
        }
        graph->CleanUp();

        delete graph;
    }
    else
    {
        std::cout << "Path does not exist: " << argv[1] << '\n';
    }

    return 0;
}
