#ifndef TOPO_H
#define TOPO_H

#include <xtensor/xarray.hpp>

#include <network.h>

xt::xarray<float> CNN(xt::xarray<float> input);

NeuralNetwork CNN2();

NeuralNetwork CNN3();

xt::xarray<float> ANN(xt::xarray<float> input);

#endif