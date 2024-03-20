#ifndef TOOLS_H
#define TOOLS_H

#include <xtensor/xarray.hpp>
#include <filesystem>

#include "network.h"

xt::xarray<float> kernelsGaussianDistro(int depth, int nbKernels, int height, int width);

xt::xarray<float> normalized(xt::xarray<float> input);

xt::xarray<float> flatten(xt::xarray<float> input);

int confirm();

std::vector<std::tuple<xt::xarray<float>, xt::xarray<float>>> loadingSet(std::string path, xt::xarray<float> label, int nbData);

std::vector<std::tuple<xt::xarray<float>, xt::xarray<float>>> loadingSets(std::string path, int nbTotalData);

void display_network(NeuralNetwork nn);

xt::xarray<float> dot_product_fma(xt::xarray<float> weights, xt::xarray<float> input);


#endif