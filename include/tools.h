#ifndef TOOLS_H
#define TOOLS_H

#include <xtensor/xarray.hpp>

xt::xarray<float> kernelsGaussianDistro(int depth, int nbKernels, int height, int width);

xt::xarray<float> flatten(xt::xarray<float> input);
#endif