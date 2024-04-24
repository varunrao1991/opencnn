#include "bmpreader.h"

#include <iostream>
#include <memory>

BITMAPFILEHEADER_POTRAIT* BmpFile::m_bitmapFileHeader = nullptr;
BITMAPINFOHEADER_POTRAIT* BmpFile::m_bitmapInfo = nullptr;

BmpFile::BmpFile(const std::string& filename) : m_filename(filename)
{
	std::cout << m_filename << std::endl;
}

void BmpFile::write_bitmap_test(unsigned char* img, uint32_t width, uint32_t height)
{
	FILE* filePtr;
	int num_comp = 3;
	if (m_bitmapFileHeader == NULL)
	{
		m_bitmapFileHeader = (BITMAPFILEHEADER_POTRAIT*)malloc(sizeof(BITMAPFILEHEADER_POTRAIT));
		m_bitmapInfo = (BITMAPINFOHEADER_POTRAIT*)malloc(sizeof(BITMAPINFOHEADER_POTRAIT));
		m_bitmapFileHeader->bfType = 19778;
		m_bitmapFileHeader->bfReserved1 = 0;
		m_bitmapFileHeader->bfReserved2 = 0;

		m_bitmapInfo->biPlanes = 1;
		m_bitmapInfo->biSize = 40;
		m_bitmapInfo->biCompression = 0;
		m_bitmapInfo->biClrImportant = 0;
		m_bitmapInfo->biClrUsed = 0;
	}
	auto error_s = fopen_s(&filePtr, m_filename.c_str(), "wb");
	if (error_s != 0)
	{
		std::cout << "Failed to write to file" << std::endl;
		return;
	}

	m_bitmapInfo->biHeight = height;
	m_bitmapInfo->biWidth = width;
	m_bitmapInfo->biBitCount = 8 * num_comp;
	m_bitmapInfo->biSizeImage = height * width * num_comp;
	if (width % 4 != 0)
	{
		m_bitmapInfo->biSizeImage += height * (width % 4);
	}
	m_bitmapFileHeader->bOffBits = 54;
	m_bitmapFileHeader->bfSize = m_bitmapInfo->biSizeImage + 54;
	if (num_comp == 1 || num_comp == 2)
	{
		m_bitmapFileHeader->bOffBits += 1024;
		m_bitmapFileHeader->bfSize += 1024;
	}
	m_bitmapInfo->biXPelsPerMeter = 0;
	m_bitmapInfo->biYPelsPerMeter = 0;

	std::fwrite(m_bitmapFileHeader, sizeof(BITMAPFILEHEADER_POTRAIT), 1, filePtr);
	std::fwrite(m_bitmapInfo, sizeof(BITMAPINFOHEADER_POTRAIT), 1, filePtr);

	if (num_comp == 1 || num_comp == 2)
	{
		for (int i = 0; i < 256; i++)
		{
			unsigned char data1 = i;
			unsigned char data2 = 0;
			std::fwrite(&data1, sizeof(unsigned char), 1, filePtr);
			std::fwrite(&data1, sizeof(unsigned char), 1, filePtr);
			std::fwrite(&data1, sizeof(unsigned char), 1, filePtr);
			std::fwrite(&data2, sizeof(unsigned char), 1, filePtr);
		}
	}
	for (int j = m_bitmapInfo->biHeight - 1; j >= 0; j--)
	{
		for (int i = 0; i < m_bitmapInfo->biWidth; i++)
		{
			uint32_t imgIndex = 3 * (j + height * i);
			unsigned char data1 = img[imgIndex];
			unsigned char data2 = img[imgIndex + 1];
			unsigned char data3 = img[imgIndex + 2];
			std::fwrite(&data3, sizeof(unsigned char), 1, filePtr);
			std::fwrite(&data2, sizeof(unsigned char), 1, filePtr);
			std::fwrite(&data1, sizeof(unsigned char), 1, filePtr);
		}
		if (m_bitmapInfo->biWidth % 4 != 0)
		{
			for (unsigned int k = 0; k < m_bitmapInfo->biWidth % 4; k++)
			{
				unsigned char data1 = 0;
				std::fwrite(&data1, sizeof(unsigned char), 1, filePtr);
			}
		}
	}
	fclose(filePtr);
	std::printf("Bitmap written : %s\n", m_filename.c_str());
	std::printf("Width : %d\n", m_bitmapInfo->biWidth);
	std::printf("Height : %d\n", m_bitmapInfo->biHeight);
}

void BmpFile::write_bitmap_char(std::vector<unsigned char>& img, uint32_t width, uint32_t height)
{
	FILE* filePtr;
	int num_comp = 3;
	if (m_bitmapFileHeader == NULL)
	{
		m_bitmapFileHeader = (BITMAPFILEHEADER_POTRAIT*)malloc(sizeof(BITMAPFILEHEADER_POTRAIT));
		m_bitmapInfo = (BITMAPINFOHEADER_POTRAIT*)malloc(sizeof(BITMAPINFOHEADER_POTRAIT));
		m_bitmapFileHeader->bfType = 19778;
		m_bitmapFileHeader->bfReserved1 = 0;
		m_bitmapFileHeader->bfReserved2 = 0;

		m_bitmapInfo->biPlanes = 1;
		m_bitmapInfo->biSize = 40;
		m_bitmapInfo->biCompression = 0;
		m_bitmapInfo->biClrImportant = 0;
		m_bitmapInfo->biClrUsed = 0;
	}
	auto error_s = fopen_s(&filePtr, m_filename.c_str(), "wb");
	if (error_s != 0)
	{
		std::cout << "Failed to write to file" << std::endl;
		return;
	}

	m_bitmapInfo->biHeight = height;
	m_bitmapInfo->biWidth = width;
	m_bitmapInfo->biBitCount = 8 * num_comp;
	m_bitmapInfo->biSizeImage = height * width * num_comp;
	if (width % 4 != 0)
	{
		m_bitmapInfo->biSizeImage += height * (width % 4);
	}
	m_bitmapFileHeader->bOffBits = 54;
	m_bitmapFileHeader->bfSize = m_bitmapInfo->biSizeImage + 54;
	if (num_comp == 1 || num_comp == 2)
	{
		m_bitmapFileHeader->bOffBits += 1024;
		m_bitmapFileHeader->bfSize += 1024;
	}
	m_bitmapInfo->biXPelsPerMeter = 0;
	m_bitmapInfo->biYPelsPerMeter = 0;

	std::fwrite(m_bitmapFileHeader, sizeof(BITMAPFILEHEADER_POTRAIT), 1, filePtr);
	std::fwrite(m_bitmapInfo, sizeof(BITMAPINFOHEADER_POTRAIT), 1, filePtr);

	if (num_comp == 1 || num_comp == 2)
	{
		for (int i = 0; i < 256; i++)
		{
			unsigned char data1 = i;
			unsigned char data2 = 0;
			std::fwrite(&data1, sizeof(unsigned char), 1, filePtr);
			std::fwrite(&data1, sizeof(unsigned char), 1, filePtr);
			std::fwrite(&data1, sizeof(unsigned char), 1, filePtr);
			std::fwrite(&data2, sizeof(unsigned char), 1, filePtr);
		}
	}
	for (int j = m_bitmapInfo->biHeight - 1; j >= 0; j--)
	{
		for (int i = 0; i < m_bitmapInfo->biWidth; i++)
		{
			uint32_t imgIndex = 3 * (j + height * i);
			unsigned char data1 = img[imgIndex];
			unsigned char data2 = img[imgIndex + 1];
			unsigned char data3 = img[imgIndex + 2];
			std::fwrite(&data3, sizeof(unsigned char), 1, filePtr);
			std::fwrite(&data2, sizeof(unsigned char), 1, filePtr);
			std::fwrite(&data1, sizeof(unsigned char), 1, filePtr);
		}
		if (m_bitmapInfo->biWidth % 4 != 0)
		{
			for (unsigned int k = 0; k < m_bitmapInfo->biWidth % 4; k++)
			{
				unsigned char data1 = 0;
				std::fwrite(&data1, sizeof(unsigned char), 1, filePtr);
			}
		}
	}
	fclose(filePtr);
	std::printf("Bitmap written : %s\n", m_filename.c_str());
	std::printf("Width : %d\n", m_bitmapInfo->biWidth);
	std::printf("Height : %d\n", m_bitmapInfo->biHeight);
}

void BmpFile::write_bitmap_file(int* img, int width, int height)
{
	FILE* filePtr;
	int channels = 3;
	int num_comp = channels;
	if (m_bitmapFileHeader == NULL)
	{
		m_bitmapFileHeader = (BITMAPFILEHEADER_POTRAIT*)malloc(sizeof(BITMAPFILEHEADER_POTRAIT));
		m_bitmapInfo = (BITMAPINFOHEADER_POTRAIT*)malloc(sizeof(BITMAPINFOHEADER_POTRAIT));
		m_bitmapFileHeader->bfType = 19778;
		m_bitmapFileHeader->bfReserved1 = 0;
		m_bitmapFileHeader->bfReserved2 = 0;

		m_bitmapInfo->biPlanes = 1;
		m_bitmapInfo->biSize = 40;
		m_bitmapInfo->biCompression = 0;
		m_bitmapInfo->biClrImportant = 0;
		m_bitmapInfo->biClrUsed = 0;
	}
	auto error_s = fopen_s(&filePtr, m_filename.c_str(), "wb");
	if (error_s != 0)
	{
		std::cout << "Failed to write to file" << std::endl;
		return;
	}

	m_bitmapInfo->biHeight = height;
	m_bitmapInfo->biWidth = width;
	m_bitmapInfo->biBitCount = 8 * channels;
	m_bitmapInfo->biSizeImage = height * width * channels;
	if (width % 4 != 0)
	{
		m_bitmapInfo->biSizeImage += height * (width % 4);
	}
	m_bitmapFileHeader->bOffBits = 54;
	m_bitmapFileHeader->bfSize = m_bitmapInfo->biSizeImage + 54;
	if (num_comp == 1 || num_comp == 2)
	{
		m_bitmapFileHeader->bOffBits += 1024;
		m_bitmapFileHeader->bfSize += 1024;
	}
	m_bitmapInfo->biXPelsPerMeter = 0;
	m_bitmapInfo->biYPelsPerMeter = 0;

	std::fwrite(m_bitmapFileHeader, sizeof(BITMAPFILEHEADER_POTRAIT), 1, filePtr);
	std::fwrite(m_bitmapInfo, sizeof(BITMAPINFOHEADER_POTRAIT), 1, filePtr);

	if (num_comp == 1 || num_comp == 2)
	{
		for (int i = 0; i < 256; i++)
		{
			unsigned char data1 = i;
			unsigned char data2 = 0;
			std::fwrite(&data1, sizeof(unsigned char), 1, filePtr);
			std::fwrite(&data1, sizeof(unsigned char), 1, filePtr);
			std::fwrite(&data1, sizeof(unsigned char), 1, filePtr);
			std::fwrite(&data2, sizeof(unsigned char), 1, filePtr);
		}
	}
	for (int j = m_bitmapInfo->biHeight - 1; j >= 0; j--)
	{
		for (int i = 0; i < m_bitmapInfo->biWidth; i++)
		{
			unsigned char data1 = img[i + width * j];
			std::fwrite(&data1, sizeof(unsigned char), 1, filePtr);
			std::fwrite(&data1, sizeof(unsigned char), 1, filePtr);
			std::fwrite(&data1, sizeof(unsigned char), 1, filePtr);
		}
		if (m_bitmapInfo->biWidth % 4 != 0)
		{
			for (unsigned int k = 0; k < m_bitmapInfo->biWidth % 4; k++)
			{
				unsigned char data1 = 0;
				std::fwrite(&data1, sizeof(unsigned char), 1, filePtr);
			}
		}
	}
	fclose(filePtr);
	std::printf("Bitmap written : %s\n", m_filename.c_str());
	std::printf("Width : %d\n", m_bitmapInfo->biWidth);
	std::printf("Height : %d\n", m_bitmapInfo->biHeight);
}

void BmpFile::write_bitmap1ch(unsigned char* img, uint32_t width, uint32_t height, uint8_t scale)
{
	FILE* filePtr;
	const int num_comp = 1;
	if (m_bitmapFileHeader == NULL)
	{
		m_bitmapFileHeader = (BITMAPFILEHEADER_POTRAIT*)malloc(sizeof(BITMAPFILEHEADER_POTRAIT));
		m_bitmapInfo = (BITMAPINFOHEADER_POTRAIT*)malloc(sizeof(BITMAPINFOHEADER_POTRAIT));
		m_bitmapFileHeader->bfType = 19778;
		m_bitmapFileHeader->bfReserved1 = 0;
		m_bitmapFileHeader->bfReserved2 = 0;

		m_bitmapInfo->biPlanes = 1;
		m_bitmapInfo->biSize = 40;
		m_bitmapInfo->biCompression = 0;
		m_bitmapInfo->biClrImportant = 0;
		m_bitmapInfo->biClrUsed = 0;
	}

	auto error_s = fopen_s(&filePtr, m_filename.c_str(), "wb");
	if (error_s != 0)
	{
		std::cout << "Failed to write to file" << std::endl;
		return;
	}

	m_bitmapInfo->biHeight = height;
	m_bitmapInfo->biWidth = width;
	m_bitmapInfo->biBitCount = 8;
	m_bitmapInfo->biSizeImage = height * width;
	if (width % 4 != 0)
	{
		m_bitmapInfo->biSizeImage += height * (width % 4);
	}
	m_bitmapFileHeader->bOffBits = 54;
	m_bitmapFileHeader->bfSize = m_bitmapInfo->biSizeImage + 54;
	if (num_comp == 1 || num_comp == 2)
	{
		m_bitmapFileHeader->bOffBits += 1024;
		m_bitmapFileHeader->bfSize += 1024;
	}
	m_bitmapInfo->biXPelsPerMeter = 0;
	m_bitmapInfo->biYPelsPerMeter = 0;

	std::fwrite(m_bitmapFileHeader, sizeof(BITMAPFILEHEADER_POTRAIT), 1, filePtr);
	std::fwrite(m_bitmapInfo, sizeof(BITMAPINFOHEADER_POTRAIT), 1, filePtr);

	if (num_comp == 1 || num_comp == 2)
	{
		for (int i = 0; i < 256; i++)
		{
			unsigned char data1 = i;
			unsigned char data2 = 0;
			std::fwrite(&data1, sizeof(unsigned char), 1, filePtr);
			std::fwrite(&data1, sizeof(unsigned char), 1, filePtr);
			std::fwrite(&data1, sizeof(unsigned char), 1, filePtr);
			std::fwrite(&data2, sizeof(unsigned char), 1, filePtr);
		}
	}

	for (int i = m_bitmapInfo->biHeight - 1; i >= 0; i--)
	{
		for (int j = 0; j < m_bitmapInfo->biWidth; j++)
		{
			int out_val = img[i * m_bitmapInfo->biWidth + j];
			unsigned char data1 = scale * out_val;
			std::fwrite(&data1, sizeof(unsigned char), 1, filePtr);
		}
		if (m_bitmapInfo->biWidth % 4 != 0)
		{
			for (unsigned int k = 0; k < m_bitmapInfo->biWidth % 4; k++)
			{
				unsigned char data1 = 0;
				std::fwrite(&data1, sizeof(unsigned char), 1, filePtr);
			}
		}
	}
	fclose(filePtr);
}

void BmpFile::write_bitmap(float* img,
	int width,
	int height,
	int channels,
	int32_t order1,
	int32_t order2,
	int32_t order3,
	float r,
	float g,
	float b,
	float factorr,
	float factorg,
	float factorb,
	float factor)
{
	FILE* filePtr;
	unsigned int i = 0, j = 0;
	int num_comp = channels;
	if (m_bitmapFileHeader == NULL)
	{
		m_bitmapFileHeader = (BITMAPFILEHEADER_POTRAIT*)malloc(sizeof(BITMAPFILEHEADER_POTRAIT));
		m_bitmapInfo = (BITMAPINFOHEADER_POTRAIT*)malloc(sizeof(BITMAPINFOHEADER_POTRAIT));
		m_bitmapFileHeader->bfType = 19778;
		m_bitmapFileHeader->bfReserved1 = 0;
		m_bitmapFileHeader->bfReserved2 = 0;

		m_bitmapInfo->biPlanes = 1;
		m_bitmapInfo->biSize = 40;
		m_bitmapInfo->biCompression = 0;
		m_bitmapInfo->biClrImportant = 0;
		m_bitmapInfo->biClrUsed = 0;
	}
	auto error_s = fopen_s(&filePtr, m_filename.c_str(), "wb");
	if (error_s != 0)
	{
		std::cout << "Failed to write to file" << std::endl;
		return;
	}

	m_bitmapInfo->biHeight = height;
	m_bitmapInfo->biWidth = width;
	m_bitmapInfo->biBitCount = 8 * channels;
	m_bitmapInfo->biSizeImage = height * width * channels;
	if (width % 4 != 0)
	{
		m_bitmapInfo->biSizeImage += height * (width % 4);
	}
	m_bitmapFileHeader->bOffBits = 54;
	m_bitmapFileHeader->bfSize = m_bitmapInfo->biSizeImage + 54;
	if (num_comp == 1 || num_comp == 2)
	{
		m_bitmapFileHeader->bOffBits += 1024;
		m_bitmapFileHeader->bfSize += 1024;
	}
	m_bitmapInfo->biXPelsPerMeter = 0;
	m_bitmapInfo->biYPelsPerMeter = 0;

	std::fwrite(m_bitmapFileHeader, sizeof(BITMAPFILEHEADER_POTRAIT), 1, filePtr);
	std::fwrite(m_bitmapInfo, sizeof(BITMAPINFOHEADER_POTRAIT), 1, filePtr);

	if (num_comp == 1 || num_comp == 2)
	{
		for (int i = 0; i < 256; i++)
		{
			unsigned char data1 = i;
			unsigned char data2 = 0;
			std::fwrite(&data1, sizeof(unsigned char), 1, filePtr);
			std::fwrite(&data1, sizeof(unsigned char), 1, filePtr);
			std::fwrite(&data1, sizeof(unsigned char), 1, filePtr);
			std::fwrite(&data2, sizeof(unsigned char), 1, filePtr);
		}
	}
	for (int j = m_bitmapInfo->biHeight - 1; j >= 0; j--)
	{
		for (int i = 0; i < m_bitmapInfo->biWidth; i++)
		{
			unsigned char data1 =
				static_cast<unsigned char>(factor * (factorr * img[3 * j + 3 * height * i + order1] + r));
			unsigned char data2 =
				static_cast<unsigned char>(factor * (factorg * img[3 * j + 3 * height * i + order2] + g));
			unsigned char data3 =
				static_cast<unsigned char>(factor * (factorb * img[3 * j + 3 * height * i + order3] + b));
			std::fwrite(&data3, sizeof(unsigned char), 1, filePtr);
			std::fwrite(&data2, sizeof(unsigned char), 1, filePtr);
			std::fwrite(&data1, sizeof(unsigned char), 1, filePtr);
		}
		if (m_bitmapInfo->biWidth % 4 != 0)
		{
			for (unsigned int k = 0; k < m_bitmapInfo->biWidth % 4; k++)
			{
				unsigned char data1 = 0;
				std::fwrite(&data1, sizeof(unsigned char), 1, filePtr);
			}
		}
	}
	fclose(filePtr);
	std::printf("Bitmap written : %s\n", m_filename.c_str());
	std::printf("Width : %d\n", m_bitmapInfo->biWidth);
	std::printf("Height : %d\n", m_bitmapInfo->biHeight);
}

std::unique_ptr<unsigned int[]> BmpFile::read_bitmap_file_unchar4(int& width, int& height)
{
	FILE* filePtr;
	int i = 0, j = 0;

	auto error_s = fopen_s(&filePtr, m_filename.c_str(), "rb");
	if (error_s != 0) {
		return std::unique_ptr<unsigned int[]>(nullptr);;
	}

	if (m_bitmapFileHeader == NULL)
		m_bitmapFileHeader = (BITMAPFILEHEADER_POTRAIT*)malloc(sizeof(BITMAPFILEHEADER_POTRAIT));
	if (m_bitmapInfo == NULL)
		m_bitmapInfo = (BITMAPINFOHEADER_POTRAIT*)malloc(sizeof(BITMAPINFOHEADER_POTRAIT));
	std::fread(m_bitmapFileHeader, sizeof(BITMAPFILEHEADER_POTRAIT), 1, filePtr);
	std::fread(m_bitmapInfo, sizeof(BITMAPINFOHEADER_POTRAIT), 1, filePtr);
	fseek(filePtr, m_bitmapFileHeader->bOffBits, SEEK_SET);

	width = m_bitmapInfo->biWidth;
	height = m_bitmapInfo->biHeight;
#if 1
	std::printf("biBitCount : %d\n", m_bitmapInfo->biBitCount);
	std::printf("biSizeImage : %d\n", m_bitmapInfo->biSizeImage);
	std::printf("Width : %d\n", m_bitmapInfo->biWidth);
	std::printf("Height : %d\n", m_bitmapInfo->biHeight);
#endif 	
	if (m_bitmapInfo->biBitCount == 32)
	{
		int num_comp = m_bitmapInfo->biSizeImage / (m_bitmapInfo->biWidth * m_bitmapInfo->biHeight);
		unsigned char* data = (unsigned char*)malloc(sizeof(unsigned char) * num_comp * m_bitmapInfo->biWidth * m_bitmapInfo->biHeight);
		std::fread(data, sizeof(unsigned char) * num_comp * m_bitmapInfo->biWidth * m_bitmapInfo->biHeight, 1, filePtr);
		for (i = 0; i < num_comp * width * height; i += num_comp) {
			unsigned char tmp = data[i];
			data[i] = data[i + 3];
			data[i + 3] = tmp;
			tmp = data[i + 1];
			data[i + 1] = data[i + 2];
			data[i + 2] = tmp;
		}
		unsigned int* data1 = (unsigned int*)data;
		for (i = 0; i < width * height / 2; i += 1)
		{
			unsigned int tmp = data1[i];
			data1[i] = data1[width * height - i];
			data1[width * height - i] = tmp;
		}
		fclose(filePtr);
		free(m_bitmapFileHeader);
		free(m_bitmapInfo);
		m_bitmapFileHeader = NULL;
		m_bitmapInfo = NULL;
		return std::unique_ptr<unsigned int[]>(data1);
	}
	else
	{
		unsigned char* data = (unsigned char*)malloc(sizeof(unsigned char) * 3 * m_bitmapInfo->biWidth * m_bitmapInfo->biHeight);
		unsigned char* data1 = (unsigned char*)malloc(sizeof(unsigned char) * 4 * m_bitmapInfo->biWidth * m_bitmapInfo->biHeight);
		fread(data, sizeof(unsigned char) * 3 * m_bitmapInfo->biWidth * m_bitmapInfo->biHeight, 1, filePtr);
		fclose(filePtr);
		unsigned int* dataref = (unsigned int*)data1;
		int cnt = 3 * width * height;
		for (i = m_bitmapInfo->biHeight - 1; i >= 0; i--) {
			for (j = 0; j < m_bitmapInfo->biWidth; j++) {
				data1[4 * j + 4 * width * i] = data[cnt - 3 * i * width + 3 * j + 0];
				data1[4 * j + 4 * width * i + 1] = data[cnt - 3 * i * width + 3 * j + 1];
				data1[4 * j + 4 * width * i + 2] = data[cnt - 3 * i * width + 3 * j + 2];
				data1[4 * j + 4 * width * i + 3] = 0;
			}
		}

		free(m_bitmapFileHeader);
		free(m_bitmapInfo);
		m_bitmapFileHeader = NULL;
		m_bitmapInfo = NULL;
		free(data);
		return std::unique_ptr<unsigned int[]>(dataref);
	}
}

std::unique_ptr<unsigned char[]> BmpFile::read_bitmap_file_unchar(int& width, int& height)
{
	FILE* filePtr;
	int i = 0, j = 0;
	auto error_s = fopen_s(&filePtr, m_filename.c_str(), "rb");
	if (error_s != 0) {
		return std::unique_ptr<unsigned char[]>(nullptr);;
	}

	if (m_bitmapFileHeader == NULL)
		m_bitmapFileHeader = (BITMAPFILEHEADER_POTRAIT*)malloc(sizeof(BITMAPFILEHEADER_POTRAIT));
	if (m_bitmapInfo == NULL)
		m_bitmapInfo = (BITMAPINFOHEADER_POTRAIT*)malloc(sizeof(BITMAPINFOHEADER_POTRAIT));
	fread(m_bitmapFileHeader, sizeof(BITMAPFILEHEADER_POTRAIT), 1, filePtr);
	fread(m_bitmapInfo, sizeof(BITMAPINFOHEADER_POTRAIT), 1, filePtr);
	fseek(filePtr, m_bitmapFileHeader->bOffBits, SEEK_SET);

	width = m_bitmapInfo->biWidth;
	height = m_bitmapInfo->biHeight;
#if 0
	std::printf("%d\n", m_bitmapInfo->biBitCount);
	std::printf("%d\n", m_bitmapInfo->biSizeImage);
	std::printf("Width  -> %d\n", m_bitmapInfo->biWidth);
	std::printf("Height -> %d\n", m_bitmapInfo->biHeight);
#endif 	
	if (m_bitmapInfo->biBitCount == 32)
	{
		int num_comp = m_bitmapInfo->biSizeImage / (m_bitmapInfo->biWidth * m_bitmapInfo->biHeight);
		unsigned char* data = (unsigned char*)malloc(sizeof(unsigned char) * num_comp * m_bitmapInfo->biWidth * m_bitmapInfo->biHeight);
		std::fread(data, sizeof(unsigned char) * num_comp * m_bitmapInfo->biWidth * m_bitmapInfo->biHeight, 1, filePtr);
		for (i = 0; i < num_comp * width * height; i += num_comp) {
			unsigned char tmp = data[i];
			data[i] = data[i + 3];
			data[i + 3] = tmp;
			tmp = data[i + 1];
			data[i + 1] = data[i + 2];
			data[i + 2] = tmp;
		}
		unsigned int* data1 = (unsigned int*)data;
		for (i = 0; i < width * height / 2; i += 1)
		{
			unsigned int tmp = data1[i];
			data1[i] = data1[width * height - i];
			data1[width * height - i] = tmp;
		}
		fclose(filePtr);
		free(m_bitmapFileHeader);
		free(m_bitmapInfo);
		m_bitmapFileHeader = NULL;
		m_bitmapInfo = NULL;
		return std::unique_ptr<unsigned char[]>(data);
	}
	else
	{
		unsigned char* data = (unsigned char*)malloc(sizeof(unsigned char) * 3 * m_bitmapInfo->biWidth * m_bitmapInfo->biHeight);
		unsigned char* data1 = (unsigned char*)malloc(sizeof(unsigned char) * 3 * m_bitmapInfo->biWidth * m_bitmapInfo->biHeight);
		fread(data, sizeof(unsigned char) * 3 * m_bitmapInfo->biWidth * m_bitmapInfo->biHeight, 1, filePtr);
		fclose(filePtr);
		int cnt = 3 * width * height - 1;
		for (i = m_bitmapInfo->biHeight - 1; i >= 0; i--) {
			for (j = 0; j < m_bitmapInfo->biWidth; j++) {
				data1[3 * i + 3 * height * (m_bitmapInfo->biWidth - 1 - j) + 2] = data[cnt - 3 * i * width - 3 * j - 2];
				data1[3 * i + 3 * height * (m_bitmapInfo->biWidth - 1 - j) + 1] = data[cnt - 3 * i * width - 3 * j - 1];
				data1[3 * i + 3 * height * (m_bitmapInfo->biWidth - 1 - j)] = data[cnt - 3 * i * width - 3 * j];
			}
		}

		free(m_bitmapFileHeader);
		free(m_bitmapInfo);
		m_bitmapFileHeader = NULL;
		m_bitmapInfo = NULL;
		free(data);
		return std::unique_ptr<unsigned char[]>(data1);
	}
}

BmpFile::~BmpFile()
= default;
