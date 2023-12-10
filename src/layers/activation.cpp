#include <iostream>

#include "activation.h"

void Activation::forward(xt::xarray<float> input)
{
    std::cout << "Activation forward" << std::endl;
}

void Activation::backward()
{
    std::cout << "Activation backward" << std::endl;
}