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
    // xt::xarray<float> input = xt::random::rand<float>({1, 100}, 5, 10);

    // Softmax1D soft{(int)input.size()};

    // std::cout << input << '\n' << std::endl;

    // soft.forward(input);

    // std::cout << soft.output << std::endl;


    CNN();

    return 0;
}
