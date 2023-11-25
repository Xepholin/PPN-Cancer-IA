#ifndef IMAGE_H
#define IMAGE_H

#include <stdint.h>

#include <xtensor/xarray.hpp>
#include <xtensor/xio.hpp>
#include <xtensor/xview.hpp>

#define SIZE_MATRICE 50

class Image
{
public:
    // Matrices with size 50x50
    xt::xarray<uint8_t> r{xt::empty<uint8_t>({SIZE_MATRICE, SIZE_MATRICE})};
    xt::xarray<uint8_t> g{xt::empty<uint8_t>({SIZE_MATRICE, SIZE_MATRICE})};
    xt::xarray<uint8_t> b{xt::empty<uint8_t>({SIZE_MATRICE, SIZE_MATRICE})};
    void saveToPNG(const char* outputPath);
};
 
std::unique_ptr<Image> pngData(const char* filename);

Image readByteFile(const char * filename, Image a);

// Convert rgb image -> grayImage (matrice)
xt::xarray<float> toGrayScale(Image a);

xt::xarray<bool> toSobel(xt::xarray<float> grayMatrice);

void saveGrayToPNG(const char* outputPath, xt::xarray<uint8_t> grayMatrice);

void saveEdgetoPBM(const char* outputPath, xt::xarray<bool> boolMatrice);

// convert les binaires en 0 ou 1, puis écrit dans le csv
void saveToCSV(const char *outputPath, const char *folderPath);

// écrit directement dans le csv sans convert
void saveToCSV2(const char *outputPath, const char *folderPath);

#endif