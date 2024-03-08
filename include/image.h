#ifndef IMAGE_H
#define IMAGE_H

#include <xtensor/xarray.hpp>
#include <xtensor/xrandom.hpp>

#include "const.h"

class Image
{
    public:
        // Matrices with size 50x50
        xt::xarray<uint8_t> r{xt::empty<uint8_t>({PNGDim, PNGDim})};
        xt::xarray<uint8_t> g{xt::empty<uint8_t>({PNGDim, PNGDim})};
        xt::xarray<uint8_t> b{xt::empty<uint8_t>({PNGDim, PNGDim})};
        
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

xt::xarray<bool> importPBM(const char *path);

xt::xarray<bool> importAllPBM(const char *path, int nbPBM);

xt::xarray<float> importAllPNG(const char *path, int nbPNG);

#endif