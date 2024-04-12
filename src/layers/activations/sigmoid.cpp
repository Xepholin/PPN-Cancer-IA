#include <iostream>
#include <cmath>

#include "sigmoid.h"

void Sigmoid::forward(xt::xarray<float> input, bool training)
{
    this->input = input;
	this->output = 1.0 / (1.0 + xt::exp(-input) );
}


xt::xarray<float> Sigmoid::backward(xt::xarray<float> cost, float learningRate)   {
	std::cout << "backward sigmoid" << std::endl;
	return 0;
}


// xt::xarray<float> Sigmoid::prime(xt::xarray<float> input) // x
// {

// 	float sig = 1.0 / (1.0+ std::exp(-x));
//     return sig *(1-sig);
// }


xt::xarray<float> Sigmoid::prime(xt::xarray<float> input) // x
{

    auto ret = 1.0 /(1.0 + xt::exp(-input) );
    return ret *(1-ret);
}

void Sigmoid::print() const
{
    std::cout << "          | Sigmoid\n"
              << "          v" << std::endl;
}


