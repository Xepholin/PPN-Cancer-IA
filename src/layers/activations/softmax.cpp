#include <iostream>
#include <cmath>

#include "softmax.h"

void Softmax::forward(xt::xarray<float> input)
{
    this->input = input;

    float expSum = xt::sum(xt::exp(this->input))();
    float maxValue = xt::amax(this->input)();

	int outputDim = this->output.shape()[0];
	float exp_xi = 0.0;

    for (int i = 0; i < outputDim; ++i)
    {
        exp_xi = std::exp(this->input(i));
        this->output(i) = std::fabs(exp_xi / expSum);
    }
}


xt::xarray<float> Softmax::backward(xt::xarray<float> cost, float learningRate)   {
	std::cout << "backward softmax" << std::endl;
	return 0;
}


float Softmax::prime(float x) // x
{
    float exp_x = std::exp(x);
    float expSum = xt::sum(xt::exp(this->input))();

    // Calculate softmax
    float softmax_x = exp_x / expSum;

    // Calculate softmax prime
    float softmax_prime_x = softmax_x * (1.0f - softmax_x);

    return softmax_prime_x;
}

void Softmax::print() const
{
    std::cout << "          | Softmax\n"
              << "          v" << std::endl;
}