#include <iostream>
#include <random>

#include "output.h"

void Output::forward(xt::xarray<float> input)
{
    this->input = input;
}

void Output::backward(xt::xarray<float> cost, float learningRate)
{
    std::cout << "backward Output" << std::endl;
}

void Output::print() const
{
    std::cout << "outputLayer: " << this->output.shape()[0] << " fully connected neurons"
              << "\n          |\n          v" << std::endl;
}