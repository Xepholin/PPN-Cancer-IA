#include <iostream>
#include <random>

#include "output.h"

void Output::forward(xt::xarray<float> input)
{
    this->input = input;

    for (int i = 0; i < this->weights.shape()[0]; ++i)
    {
        for (int j = 0; j < this->weights.shape()[1]; ++j)
        {
            if (drop(i, j) & 0)
            {
                continue;
            }
            output(j) += this->weights(i, j) * this->input(i);
        }
    }
}

void Output::backward(xt::xarray<float> gradient, float learningRate)
{
    std::cout << "backward Output" << std::endl;
}

void Output::print() const
{
    std::cout << "outputLayer: " << this->output.shape()[0] << " fully connected neurons"
              << "\n          |\n          v" << std::endl;
}