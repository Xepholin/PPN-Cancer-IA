#include <iostream>
#include <cmath>

#include "sigmoid.h"

void Sigmoid::forward(xt::xarray<float> input, bool training __attribute__((unused)))
{
    this->input = input;
	this->output = 1.0 / (1.0 + xt::exp(-input) );
}


xt::xarray<float> Sigmoid::backward(xt::xarray<float> cost __attribute__((unused)))   {
	std::cout << "backward sigmoid" << std::endl;
	return 0;
}

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


