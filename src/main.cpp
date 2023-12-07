#include <iostream>
#include <tuple>

#include "image.h"
#include "convolution.h"
#include "network.h"
#include "activations.h"

#include "tools.h"

int main()
{
    PoolingLayer pool{2, 2, 0, PoolingLayer::PoolingType::AVG};
    ReLu relu;

    std::tuple<int, int, int> inputShape{6, 6, 6};
    std::tuple<int, int, int, int, int> filtersShape{4, 2, 2, 1, 0};

    auto input = xt::random::rand<float>({ 6,6,6}, 0, 1);  

    ConvolutionLayer conv{6, inputShape, filtersShape, relu, pool};

    std::cout << input << "\n" << std::endl;

    std::cout << conv.filters << "\n" << std::endl;
    std::cout << conv.filters.shape()[0] << "\n" << std::endl;
    std::cout << conv.filters.shape()[1] << "\n" << std::endl;
    std::cout << conv.filters.shape()[2] << "\n" << std::endl;
    std::cout << conv.filters.shape()[3] << "\n" << std::endl;


    std::cout << "Input shape : \n" << std::endl;
    std::cout << conv.input.shape()[0] << "\n" << std::endl;
    std::cout << conv.input.shape()[1] << "\n" << std::endl;
    std::cout << conv.input.shape()[2] << "\n" << std::endl;

    conv.forward(input);
    std::cout << "Output shape : \n" << std::endl;
    std::cout << conv.output.shape()[0] << "\n" << std::endl;
    std::cout << conv.output.shape()[1] << "\n" << std::endl;
    std::cout << conv.output.shape()[2] << "\n" << std::endl;

    std::cout << conv.output << "\n" << std::endl;

    
    ReLu abc {};
    abc.output = xt::empty<float>({4,5,5});
    abc.forward(conv.output);
    std::cout << abc.output << "\n" << std::endl;


    std::cout << abc.output.shape()[0] << "\n" << std::endl;
    std::cout << abc.output.shape()[1] << "\n" << std::endl;
    std::cout << abc.output.shape()[2] << "\n" << std::endl;

    // std::vector <std::vector <std::vector <xt::xarray<float>>>> a ; 

    return 0;
}
