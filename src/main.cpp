#include <iostream>
#include <tuple>

#include "image.h"
#include "convolution.h"
#include "network.h"

#include "tools.h"

int main()
{
    Pooling pool{3, 3, 0, Pooling::PoolingType::AVG};

    std::tuple<int, int, int> inputShape{4, 4, 3};
    std::tuple<int, int, int, int, int> filtersShape{2, 2, 4, 1, 0};

    ConvolutionLayer conv(6, inputShape, filtersShape, pool);

    // std::cout << conv.filters << std::endl;

    conv.forward(xt::xarray<float>({{1, 1, 1}, {1, 1, 1}}));
    conv.backward(xt::xarray<float>({{1, 1, 1}, {1, 1, 1}}));

    xt::xarray<float> test{{1, 2, 3, 1, 2, 3},
                           {4, 5, 6, 4, 5, 6},
                           {7, 8, 9, 7, 8, 9},
                           {1, 2, 3, 1, 2, 3},
                           {4, 5, 6, 4, 5, 6},
                           {7, 8, 9, 7, 8, 9}};

    auto a = conv.poolingMatrice(test);

    std::cout << test << std::endl;
    std::cout << "Pooled Matrice avec size  " <<  pool.size << "et stride  " <<  pool.stride<< std::endl;
    std::cout << a << std::endl;

    return 0;
}
