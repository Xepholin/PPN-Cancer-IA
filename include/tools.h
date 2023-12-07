#ifndef TOOLS_H
#define TOOLS_H

#include <xtensor/xarray.hpp>

xt::xarray<bool> importPBM(const char *path);
xt::xarray<bool> importAllPBM(const char *path, int nbPBM);
xt::xarray<float> kernelsGaussianDistro(int depth, int nbKernels, int height, int width);

#endif