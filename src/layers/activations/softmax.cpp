#include <iostream>
#include <cmath>

#include "softmax.h"

void Softmax::forward(xt::xarray<float> input)
{
    this->input = input;

    float expSum = xt::sum(xt::exp(this->input))();
    float maxValue = xt::amax(this->input)();

    for (int i = 0; i < this->output.shape()[0]; ++i)
    {
        // auto exp_xi = std::exp(this->input(i) - maxValue);
        float exp_xi = std::exp(this->input(i));
        this->output(i) = std::abs(exp_xi / expSum);
    }
}


xt::xarray<float> Softmax::backward(xt::xarray<float> cost, float learningRate)   {
	std::cout << "backward softmax" << std::endl;
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

xt::xarray<float> Softmax::softmaxJacobien()
{
    xt::xarray<float> expOutput = xt::exp(this->output);
    float sum = xt::sum(expOutput)();
    xt::xarray<float> jacobien = xt::empty<float> ({this->output.size(), this->output.size()});


    for(int i = 0; i < jacobien.shape()[0]; ++i)
    {
        for(int j = 0; j < jacobien.shape()[1]; ++j)
        {
            if (i == j)
            {
                jacobien(i, j) += (expOutput(i)/sum)*(1 - expOutput(j)/sum);
            }

            else
            {
                jacobien(i, j) += -(expOutput(i)/sum)*(expOutput(j)/sum);
            }
        }
    }

    return jacobien;
}


xt::xarray<float> Softmax::softmaxGradient()
{

    int num_classes = this->output.shape()[0];

    // Compute the Jacobian matrix of the softmax function
    xt::xarray<float> jacobian = this->output * xt::transpose(1.0 - this->output);

    // Compute the gradient of the softmax output with respect to the input
    xt::xarray<float> gradient = xt::empty<float>({this->output.shape()[0]});

    for (int i = 0; i < num_classes; ++i)
    {
        // gradient(i) = xt::sum(jacobian, {-1});
    }

    return this->output;
}
