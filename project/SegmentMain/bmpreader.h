
#pragma once

#include <memory>
#include <string>
#include <vector>

#pragma pack(push, 1)
typedef struct tagBITMAPFILEHEADER_
{
	unsigned short bfType;
	unsigned int bfSize;
	unsigned short bfReserved1;
	unsigned short bfReserved2;
	unsigned int bOffBits;
}BITMAPFILEHEADER_POTRAIT;

typedef struct tagBITMAPINFOHEADER_
{
	unsigned int biSize;
	unsigned int  biWidth;
	unsigned int  biHeight;
	unsigned short biPlanes; 	unsigned short biBitCount; 	unsigned int  biCompression;	unsigned int  biSizeImage;
	unsigned int  biXPelsPerMeter;
	unsigned int  biYPelsPerMeter;
	unsigned int  biClrUsed;
	unsigned int  biClrImportant;
}BITMAPINFOHEADER_POTRAIT;
#pragma pack(pop)

class BmpFile
{
public:
	explicit BmpFile(const std::string& filename);
	virtual ~BmpFile();
	std::unique_ptr<unsigned char[]> read_bitmap_file_unchar(int& width, int& height);
	std::unique_ptr<unsigned int[]> read_bitmap_file_unchar4(int& width, int& height);
	void write_bitmap_test(unsigned char* img, uint32_t width, uint32_t height);
	void write_bitmap_char(std::vector<unsigned char>& img, uint32_t width, uint32_t height);
	void write_bitmap_file(int* img, int width, int height);
	void write_bitmap1ch(unsigned char* img, uint32_t width, uint32_t height, uint8_t scale);
	void write_bitmap(float* img, int width, int height, int channels, int32_t order1 = 0, int32_t order2 = 1, int32_t order3 = 2, float r = 0, float g = 0, float b = 0, float factorr = 1, float factorg = 1, float factorb = 1, float factor = 1);

private:
	std::string m_filename;
	static BITMAPFILEHEADER_POTRAIT* m_bitmapFileHeader;
	static BITMAPINFOHEADER_POTRAIT* m_bitmapInfo;
};

