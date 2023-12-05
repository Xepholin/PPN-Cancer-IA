#include <iostream>
#include <tuple>

#include "image.h"
#include "convolution.h"
#include "network.h"

#include "tools.h"

int main()
{
    Pooling pool{Pooling::PoolingType::NO_TYPE,  0 , 1};
    
    std::tuple<int, int, int> inputShape{4, 4, 3};
    std::tuple<int, int, int,int,int> filtersShape{2, 2, 4, 1, 0};

    ConvolutionLayer conv(6, inputShape, filtersShape, pool);  // Assuming RGB images with a 3x3 kernel and 64 filters

    std::cout << conv.filters << std::endl;

    conv.forward(xt::xarray<float>({{1, 1, 1}, {1, 1, 1}}));
    conv.backward(xt::xarray<float>({{1, 1, 1}, {1, 1, 1}}));

    

    return 0;
}
