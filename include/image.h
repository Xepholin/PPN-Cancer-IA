#ifndef IMAGE_H
#define IMAGE_H

#include <stdint.h>

#include <xtensor/xarray.hpp>
#include <xtensor/xio.hpp>
#include <xtensor/xview.hpp>

#define PRE_SIZE_MATRIX 50

class Image
{
public:
    // Matrices with size 50x50
    xt::xarray<uint8_t> r{xt::empty<uint8_t>({PRE_SIZE_MATRIX, PRE_SIZE_MATRIX})};
    xt::xarray<uint8_t> g{xt::empty<uint8_t>({PRE_SIZE_MATRIX, PRE_SIZE_MATRIX})};
    xt::xarray<uint8_t> b{xt::empty<uint8_t>({PRE_SIZE_MATRIX, PRE_SIZE_MATRIX})};
    void saveToPNG(const char* outputPath);
};
 
std::unique_ptr<Image> pngData(const char* filename);

Image readByteFile(const char * filename, Image a);

// Convert rgb image -> grayImage (matrice)
xt::xarray<float> toGrayScale(Image a);

xt::xarray<bool> toSobel(xt::xarray<float> grayMatrice);

void saveGrayToPNG(const char* outputPath, xt::xarray<uint8_t> grayMatrice);

void saveEdgetoPBM(const char* outputPath, xt::xarray<bool> boolMatrice);

void generateAllPBM(const char *folderConvPath, const char *folderOutput);

// Avec C++17
void generateAllPBM2(const char *folderConvPath, const char *folderOutput);

#endif