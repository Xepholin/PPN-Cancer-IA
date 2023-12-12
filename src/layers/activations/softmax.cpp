#include <iostream>
#include <cmath>

#include <xtensor/xio.hpp>

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
        this->output(i) = exp_xi / expSum;
    }
}

void Softmax::backward(xt::xarray<float> cost, float learningRate)   {

    xt::xarray<float> softmax_grad = softmaxGradient();

    xt::xarray<float> grad_input = cost * softmax_grad;

    this->input -= learningRate * grad_input;
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
