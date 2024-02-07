#ifndef TOOLS_H
#define TOOLS_H

#include <xtensor/xarray.hpp>
#include <filesystem>

#include "network.h"

xt::xarray<float> kernelsGaussianDistro(int depth, int nbKernels, int height, int width);

xt::xarray<float> normalized(xt::xarray<float> input);

xt::xarray<float> flatten(xt::xarray<float> input);

float MSE(xt::xarray<float> output, xt::xarray<float> trueValue);

float crossEntropy(xt::xarray<float> output, xt::xarray<int> trueValue);

int continueTraining();

void saveConfirm(NeuralNetwork nn, bool loaded);

void display_network(NeuralNetwork nn);

#endif