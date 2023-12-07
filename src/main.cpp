#include <iostream>
#include <tuple>

#include "image.h"
#include "convolution.h"
#include "network.h"
#include "activations.h"

#include "tools.h"

int main()
{
    std::tuple<int, int, int> inputShape{6, 6, 6};
    std::tuple<int, int, int, int, int> filtersShape{4, 2, 2, 1, 0};

    auto input = xt::random::rand<float>({6,6,6}, 0, 1);  

    ConvolutionLayer conv{6, inputShape, filtersShape};

    std::cout << input << "\n" << std::endl;

    conv.forward(input);

    std::cout << conv.output << "\n" << std::endl;

    ReLu abc {conv.outputShape};

    abc.forward(conv.output);
    std::cout << abc.output << "\n" << std::endl;

    PoolingLayer::PoolingType type = PoolingLayer::PoolingType::MAX;

    PoolingLayer pool{abc.outputShape, 3, 1, 0, type};

    pool.forward(abc.output);
    std::cout << pool.output << std::endl;
    std::cout << pool.output.shape()[0] << std::endl;
    std::cout << pool.output.shape()[1] << std::endl;
    std::cout << pool.output.shape()[2] << std::endl;

    // std::vector <std::vector <std::vector <xt::xarray<float>>>> a ; 

    return 0;
}
