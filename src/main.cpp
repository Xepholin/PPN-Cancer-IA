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

    //xt::xarray<float> input = xt::random::randn<float>({1, 1, 1});
    xt::xarray<float> input = xt::zeros<float>({1, 1, 100});

    srand(time(NULL));

    for (int i = 0; i < 100; ++i)    {
        float r = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
        input(i) = r;
    }

    auto prediction = ANN(input);

    std::cout << prediction << std::endl;



    return 0;
}
