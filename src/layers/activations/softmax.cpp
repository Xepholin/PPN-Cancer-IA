#include <iostream>
#include <cmath>

#include "softmax.h"

void Softmax1D::forward(xt::xarray<float> input)
{
    this->input = input;

    float expSum = xt::sum(xt::exp(this->input))();
    float maxValue = xt::amax(this->input)();

    for (int i = 0; i < this->output.shape()[0]; ++i)
    {
        // auto exp_xi = std::exp(this->input(i) - maxValue);
        float exp_xi = std::exp(this->input(i) );
        this->output(i) = exp_xi / expSum;
    }

    std::cout << "          | Softmax\n"
              << "          v" << std::endl;
}

void Softmax1D::backward(xt::xarray<float> gradient)
{
    std::cout << "Softmax backward" << std::endl;
}