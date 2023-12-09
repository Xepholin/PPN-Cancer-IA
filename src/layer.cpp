#include <iostream>

#include "layer.h"

void ILayer::forward(xt::xarray<float> input)
{
    std::cout << "ILayer forward" << std::endl;
}

void ILayer::backward(xt::xarray<float> gradient)
{
    std::cout << "ILayer backward" << std::endl;
}