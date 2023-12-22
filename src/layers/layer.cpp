#include <iostream>

#include "layer.h"

void ILayer::forward(xt::xarray<float> input)
{
    std::cout << "forward ILayer" << std::endl;
}

void ILayer::backward(xt::xarray<float> cost, float learningRate)
{
    std::cout << "backward ILayer" << std::endl;
}

void ILayer::print() const
{
    std::cout << "print ILayer" << std::endl; 
}