#include "activations.h"

void ReLu::forward(xt::xarray<float> input)
{
    std::cout << "ReLu forward" << std::endl;
}

void ReLu::backward(xt::xarray<float> gradient)
{
    std::cout << "ReLu backward" << std::endl;
}

float ReLu::activation(xt::xarray<float> matrix)
{
    std::cout << "Ici ca active ReLu" << std::endl;
    return 0.0;
}