#ifndef TOPO_H
#define TOPO_H

#include <xtensor/xarray.hpp>

#include <network.h>

NeuralNetwork CNN();

NeuralNetwork CNN2(std::tuple<int, int, int> inputShape);

NeuralNetwork CNN3(std::tuple<int, int, int> inputShape);

#endif