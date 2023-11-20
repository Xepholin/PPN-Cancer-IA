#ifndef IMAGE_H
#define IMAGE_H

#include <stdint.h>

#include <xtensor/xarray.hpp>
#include <xtensor/xio.hpp>
#include <xtensor/xrandom.hpp>

class Image
{
public:
    // Matrices with size 50x50
    xt::xarray<uint8_t> r{xt::empty<uint8_t>({50, 50})};
    xt::xarray<uint8_t> g{xt::empty<uint8_t>({50, 50})};
    xt::xarray<uint8_t> b{xt::empty<uint8_t>({50, 50})};
    void toGrayscale();
    void saveToPNG(const char* outputPath);
};

std::unique_ptr<Image> pngData(const char* filename);
Image readByteFile(const char * filename, Image a);


#endif