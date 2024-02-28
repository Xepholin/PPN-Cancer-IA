#include <iostream>
#include <cmath>

#include "sigmoid.h"

void Sigmoid::forward(xt::xarray<float> input)
{
    this->input = input;
	this->output = 1.0 / (1.0+ xt::exp(-input) );
}


xt::xarray<float> Sigmoid::backward(xt::xarray<float> cost, float learningRate)   {
	std::cout << "backward sigmoid" << std::endl;
}


float Sigmoid::prime(float x) // x
{
	float sig = 1.0 / (1.0+ std::exp(-x));
    return sig *(1-sig);
}

void Sigmoid::print() const
{
    std::cout << "          | Sigmoid\n"
              << "          v" << std::endl;
}


