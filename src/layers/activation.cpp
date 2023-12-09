#include <iostream>

#include "activation.h"

void Activation::forward(xt::xarray<float> input)
{
    std::cout << "Activation forward" << std::endl;
}

void Activation::backward(xt::xarray<float> gradient)
{
    std::cout << "Activation backward" << std::endl;
}

xt::xarray<float> Activation::activation(xt::xarray<float> matrix)
{
    std::cout << "Ici ca active Convolution" << std::endl;
    return 0.0;
}