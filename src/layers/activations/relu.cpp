#include <iostream>

#include "relu.h"

void ReLu::forward(xt::xarray<float> input)
{
    this->output = xt::where(input <= 0, 0.0, input);
}

void ReLu::backward(xt::xarray<float> cost, float learningRate)
{
    std::cout << "ReLu backward" << std::endl;
}

float ReLu::prime(float x)   {
    if (x >= 0.0)
    {
        return 1.0;
    }
    return 0.0;
}

void ReLu::print() const
{
    std::cout << "          | ReLu\n"
              << "          v" << std::endl;
}