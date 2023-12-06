#include <iostream>
#include <tuple>

#include "image.h"
#include "convolution.h"
#include "network.h"

#include "tools.h"

int main()
{
    PoolingLayer pool(3, 3, 0, PoolingLayer::PoolingType::AVG);

    std::tuple<int, int, int> inputShape{4, 4, 3};
    std::tuple<int, int, int, int, int> filtersShape{2, 2, 4, 1, 0};

    ConvolutionLayer conv(6, inputShape, filtersShape);

    conv.forward(xt::xarray<float>({{1, 1, 1}, {1, 1, 1}}));
    conv.backward(xt::xarray<float>({{1, 1, 1}, {1, 1, 1}}));

    xt::xarray<float> test{{1, 2, 3, 1, 2, 3},
                           {4, 5, 6, 4, 5, 6},
                           {7, 8, 9, 7, 8, 9},
                           {1, 2, 3, 1, 2, 3},
                           {4, 5, 6, 4, 5, 6},
                           {7, 8, 9, 7, 8, 9}};

    auto a = pool.poolingMatrice(test);

    std::cout << test << std::endl;
    std::cout << "Pooled Matrice avec size  " <<  pool.size << "et stride  " <<  pool.stride<< std::endl;
    std::cout << a << std::endl;

    return 0;
}
