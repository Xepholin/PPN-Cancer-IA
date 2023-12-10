#include <iostream>
#include <tuple>

#include <xtensor/xarray.hpp>
#include <xtensor/xio.hpp>
#include <xtensor/xrandom.hpp>

#include "image.h"
#include "tools.h"
#include "conv_op.h"

#include "convolution.h"
#include "activation.h"
#include "relu.h"
#include "softmax.h"
#include "pooling.h"
#include "dense.h"

#include "nn.h"

int main()
{
    xt::xarray<float> input = xt::random::randn<float>({1, 5});

    // Softmax1D soft{(int)input.size()};

    // std::cout << input << '\n' << std::endl;

    // soft.forward(input);

    // std::cout << soft.output << std::endl;

    xt::xarray<float> image = importPBM("../assets/PBM/8863_idx5_x451_y551_class0.pbm");
    image.reshape({1, 48, 48});

    std::cout << image << std::endl;

    CNN(image);

    return 0;
}
