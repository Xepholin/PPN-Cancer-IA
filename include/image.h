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


// Convert rgb image -> grayImage (matrice)
xt::xarray<float> toGrayScale(Image a);

void saveGrayToPNG(const char* outputPath, xt::xarray<uint8_t> grayMatrice);
 
std::unique_ptr<Image> pngData(const char* filename);

Image readByteFile(const char * filename, Image a);

void saveEdgetoPBM(const char* outputPath, xt::xarray<bool> boolMatrice ) ;



#endif