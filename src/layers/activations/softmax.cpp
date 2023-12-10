#include <iostream>
#include <cmath>

#include "softmax.h"

void Softmax2D::forward(xt::xarray<float> input)
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

    std::cout << "          | Softmax\n"
              << "          v" << std::endl;
}

void Softmax2D::backward()
{
    // Compute the Jacobian matrix of the softmax function
    auto jacobian = -this->output * xt::transpose(this->output);
    auto gradient = softmaxGradient();
    // Compute the gradient of the loss with respect to the softmax input
    this->input = xt::sum(gradient * jacobian, {-1});
}

xt::xarray<float> Softmax2D::softmaxGradient()
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