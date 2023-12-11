#include <iostream>
#include <tuple>

#include <xtensor/xarray.hpp>
#include <xtensor/xio.hpp>
#include <xtensor/xrandom.hpp>

#include "image.h"
#include "tools.h"
#include "conv_op.h"
#include "layer.h"

#include "convolution.h"
#include "activation.h"
#include "relu.h"
#include "softmax.h"
#include "pooling.h"
#include "dense.h"

#include "nn.h"

int main()
{   
    std::random_device rd;
    std::mt19937 gen(rd());

    xt::xarray<float> prediction = xt::empty<float>({5, 2});

    for (int i = 0; i < 1; ++i) {
        xt::xarray<float> input = xt::zeros<float>({1, 48, 48});

        for (int j = 0; j < 48; ++j)
        {
            for (int k = 0; k < 48; ++k)
            {
                input(j, k) = std::uniform_int_distribution<>(0, 1)(gen);
            }
        }

        xt::view(prediction, i) = CNN2(input);
    }

    std::cout << prediction << std::endl;

    return 0;
}
