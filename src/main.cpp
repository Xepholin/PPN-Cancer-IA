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
#include "pooling.h"
#include "dense.h"

#include "nn.h"

int main()
{
    // xt::xarray<float> input = xt::random::randint({3,3,3}, 0, 10);
    // auto input2 = input;
    // // xt::xarray<float> input = xt::random::randint({2,3, 5, 5}, -10, 10);  
    // std::cout << input << '\n' << std::endl;
    // // std::cout << xt::where(input <= 0, 0.0, input) << std::endl;
    
    // std::cout << input << '\n' << std::endl;

    CNN();

    return 0;
}
