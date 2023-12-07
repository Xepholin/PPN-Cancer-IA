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

    std::tuple<int, int, int> inputShape{6, 44, 44};
    std::tuple<int, int, int, int, int> filtersShape{2, 2, 4, 1, 0};

    ConvolutionLayer conv{6, inputShape, filtersShape, relu, pool};

    // conv.forward(xt::xarray<float>({{1, 1, 1}, {1, 1, 1}}));
    // conv.backward(xt::xarray<float>({{1, 1, 1}, {1, 1, 1}}));

    // xt::xarray<float> test{{1, 2, 3, 1, 2, 3},
    //                        {4, 5, 6, 4, 5, 6},
    //                        {7, 8, 9, 7, 8, 9},
    //                        {1, 2, 3, 1, 2, 3},
    //                        {4, 5, 6, 4, 5, 6},
    //                        {7, 8, 9, 7, 8, 9}};

    // auto a = pool.poolingMatrice(test);

    xt::xarray<float> input = xt::random::rand<float>({3, 4, 4}, -1, 1);

    // auto abcd = pool.poolingMatrice(input);

    // std::cout << abcd.shape()[0] << "\n" << std::endl;
    // std::cout << abcd.shape()[1] << "\n" << std::endl;
    std::cout << input << "\n" << std::endl;


    ReLu abc {};
    abc.output = xt::empty<float>({3, 4, 4});

    abc.forward(input);
    
    std::cout << abc.output << "\n" << std::endl;

    // comment on peut tester cette merde
    // je sais

    // std::vector <std::vector <xt::xarray<float>>> input;

    // std::vector <std::vector <std::vector <xt::xarray<float>>>> a ; 

    return 0;
}
