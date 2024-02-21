#ifndef TOPO_H
#define TOPO_H

#include <xtensor/xarray.hpp>

#include <network.h>

NeuralNetwork CNN();

NeuralNetwork CNN2(std::tuple<int, int, int> inputShape, std::string name, float learningRate);

NeuralNetwork CNN3(std::tuple<int, int, int> inputShape, std::string name, float learningRate);

#endif