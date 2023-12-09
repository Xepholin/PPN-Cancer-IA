#include <iostream>

#include <tuple>

#include <xtensor/xarray.hpp>
#include <xtensor/xrandom.hpp>
#include <xtensor/xio.hpp>

#include "convolution.h"
#include "activation.h"
#include "relu.h"
#include "softmax.h"
#include "pooling.h"
#include "dense.h"

void CNN()
{
    
    std::cout << "Convolutional Neural Network\n" << "---------------" << std::endl;

    int fil1 = 32;
    int fil2 = 64;
    int fil3 = 128;
    
    xt::xarray<float> input = xt::random::randint({1,48, 48},0,1);  

    // ------------------------------------------------------------------------------

    std::tuple<int, int, int> conv1_inputShape{1, 48, 48};
    std::tuple<int, int, int, int, int> conv1_filtersShape{fil1, 5, 5, 1, 0};
    Convolution conv1{1, conv1_inputShape, conv1_filtersShape};

    conv1.forward(input) ;  // 32 x 44 x 44

    ReLu3D relu3d_1{conv1.outputShape};
    relu3d_1.forward(conv1.output);

    Pooling pool_1{relu3d_1.outputShape,2,2,0,Pooling::MAX};
    pool_1.forward(relu3d_1.output);    // 32 x 22 x 22

    // ------------------------------------------------------------------------------

    std::tuple<int, int, int, int, int> conv2_filtersShape{fil2, 3, 3, 1, 0};
    Convolution conv2{std::get<0>(pool_1.outputShape), pool_1.outputShape, conv2_filtersShape};

    conv2.forward(pool_1.output);

    ReLu3D relu3d_2{conv2.outputShape};
    relu3d_2.forward(conv2.output);

    Pooling pool_2{relu3d_2.outputShape,2,2,0,Pooling::MAX};
    pool_2.forward(relu3d_2.output);

    // ------------------------------------------------------------------------------

    std::tuple<int, int, int, int, int> conv3_filtersShape{fil2, 3, 3, 1, 0};
    Convolution conv3{std::get<0>(pool_2.outputShape), pool_2.outputShape, conv3_filtersShape};

    conv3.forward(pool_2.output);

    ReLu3D relu3d_3{conv3.outputShape};
    relu3d_3.forward(conv3.output);


    // ------------------------------------------------------------------------------

    xt::xarray<float> flatted = flatten(relu3d_3.output);
    int flattedSize = flatted.size();
    
    // ------------------------------------------------------------------------------

    Dense dense1{flattedSize, 2048};

    dense1.forward(flatted);

    ReLu1D relu1D_1{dense1.outputShape};

    relu1D_1.forward(dense1.output);

    dense1.dropout(50);

    // ------------------------------------------------------------------------------

    Dense dense2{relu1D_1.outputShape, 2048};

    dense2.forward(relu1D_1.output);

    ReLu1D relu1D_2{dense2.outputShape};

    relu1D_2.forward(dense2.output);

    dense2.dropout(50);

    // ------------------------------------------------------------------------------
    
    Dense dense3{relu1D_2.outputShape, 2};

    dense3.forward(relu1D_2.output);

    Softmax1D soft_1{dense3.outputShape};

    soft_1.forward(dense3.output);

    // ------------------------------------------------------------------------------

    std::cout << "\nPrédiction\n" << "---------------" << std::endl;

    std::cout << soft_1.output << std::endl;
}