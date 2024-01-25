#ifndef TOPO_H
#define TOPO_H

#include <xtensor/xarray.hpp>

#include <network.h>

NeuralNetwork CNN();

NeuralNetwork CNN2(std::tuple<int, int, int> inputShape, float learningRate, u_int16_t dropRate);

NeuralNetwork CNN3(std::tuple<int, int, int> inputShape, float learningRate, u_int16_t dropRate);

#endif