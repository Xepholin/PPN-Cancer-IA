#include <iostream>

#include "network.h"

void ILayer::forward(xt::xarray<float> input)  {
    std::cout << "ILayer forward" << std::endl;
}

void ILayer::backward(xt::xarray<float> gradient) {
    std::cout << "ILayer backward" << std::endl;
}
 
//

void Convolution::forward(xt::xarray<float> input) {
    std::cout << "Convolution forward" << std::endl;
}

void Convolution::backward(xt::xarray<float> gradient)    {
    std::cout << "Convolution backward" << std::endl;
}