#include <iostream>
#include <random>

#include "outputlayer.h"
#include "outputlayer.h"

void outputLayer::forward(xt::xarray<float> input)
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

    std::cout << "outputLayer: " << this->output.shape()[0] << " fully connected neurons"
              << "\n          |\n          v" << std::endl;
}