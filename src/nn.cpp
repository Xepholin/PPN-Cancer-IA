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

xt::xarray<float> CNN(xt::xarray<float> input)
{
    // std::cout << "Convolutional Neural Network\n" << "---------------" << std::endl;

    int fil1 = 32;
    int fil2 = 64;
    int fil3 = 128;

    // ------------------------------------------------------------------------------

    std::tuple<int, int, int> conv1_inputShape{1, 48, 48};
    std::tuple<int, int, int, int, int> conv1_filtersShape{fil1, 5, 5, 1, 0};

    // ------------------------------------------------------------------------------

    Convolution conv1{1, conv1_inputShape, conv1_filtersShape};

    ReLu3D relu3d_1{conv1.outputShape};

    Pooling pool_1{relu3d_1.outputShape,2,2,0,PoolingType::MAX};

    // ------------------------------------------------------------------------------

    std::tuple<int, int, int, int, int> conv2_filtersShape{fil2, 3, 3, 1, 0};
    Convolution conv2{std::get<0>(pool_1.outputShape), pool_1.outputShape, conv2_filtersShape};

    ReLu3D relu3d_2{conv2.outputShape};

    Pooling pool_2{relu3d_2.outputShape,2,2,0,PoolingType::MAX};

    // ------------------------------------------------------------------------------

    std::tuple<int, int, int, int, int> conv3_filtersShape{fil2, 3, 3, 1, 0};
    Convolution conv3{std::get<0>(pool_2.outputShape), pool_2.outputShape, conv3_filtersShape};

    ReLu3D relu3d_3{conv3.outputShape};

    xt::xarray<float> flatted = flatten(relu3d_3.output);
    int flattedSize = flatted.size();

    // ------------------------------------------------------------------------------

    Dense dense1{flattedSize, 2048};

    ReLu1D relu1D_1{dense1.outputShape};

    // ------------------------------------------------------------------------------

    Dense dense2{relu1D_1.outputShape, 2048};

    ReLu1D relu1D_2{dense2.outputShape};

    // ------------------------------------------------------------------------------

    Dense dense3{relu1D_2.outputShape, 1024};

    ReLu1D relu1D_3{dense3.outputShape};

    // ------------------------------------------------------------------------------

    Dense dense4{relu1D_2.outputShape, 512};

    ReLu1D relu1D_4{dense4.outputShape};

    // ------------------------------------------------------------------------------

    Dense dense5{relu1D_4.outputShape, 2};

    Softmax1D soft_1{dense5.outputShape};

    // ------------------------------------------------------------------------------

    conv1.forward(input) ;  // 32 x 44 x 44

    relu3d_1.forward(conv1.output);

    pool_1.forward(relu3d_1.output);    // 32 x 22 x 22

    // ------------------------------------------------------------------------------

    conv2.forward(pool_1.output);

    relu3d_2.forward(conv2.output);

    pool_2.forward(relu3d_2.output);

    // ------------------------------------------------------------------------------

    conv3.forward(pool_2.output);

    relu3d_3.forward(conv3.output);
    
    // ------------------------------------------------------------------------------

    dense1.dropout(50);

    dense1.forward(flatted);

    relu1D_1.forward(dense1.output);

    // ------------------------------------------------------------------------------

    dense2.dropout(50);

    dense2.forward(relu1D_1.output);

    relu1D_2.forward(dense2.output);

    // ------------------------------------------------------------------------------

    dense3.dropout(50);

    dense3.forward(relu1D_2.output);

    relu1D_3.forward(dense3.output);

    // ------------------------------------------------------------------------------

    dense4.dropout(50);

    dense4.forward(relu1D_3.output);

    relu1D_4.forward(dense4.output);

    // ------------------------------------------------------------------------------

    dense5.dropout(50);

    dense5.forward(relu1D_4.output);

    soft_1.forward(dense5.output);

    return soft_1.output;
}

xt::xarray<float> CNN2(xt::xarray<float> input)
{
    int fil1 = 32;
    int fil2 = 64;
    int fil3 = 128;

    // ------------------------------------------------------------------------------

    std::tuple<int, int, int> conv1_inputShape{1, 48, 48};
    std::tuple<int, int, int, int, int> conv1_filtersShape{fil1, 5, 5, 1, 0};

    // ------------------------------------------------------------------------------

    Convolution conv1{1, conv1_inputShape, conv1_filtersShape};

    ReLu3D relu3d_1{conv1.outputShape};

    Pooling pool_1{relu3d_1.outputShape,2,2,0,PoolingType::MAX};

    // ------------------------------------------------------------------------------

    std::tuple<int, int, int, int, int> conv2_filtersShape{fil2, 3, 3, 1, 0};
    Convolution conv2{std::get<0>(pool_1.outputShape), pool_1.outputShape, conv2_filtersShape};

    ReLu3D relu3d_2{conv2.outputShape};

    Pooling pool_2{relu3d_2.outputShape,2,2,0,PoolingType::MAX};

    // ------------------------------------------------------------------------------

    std::tuple<int, int, int, int, int> conv3_filtersShape{fil2, 3, 3, 1, 0};
    Convolution conv3{std::get<0>(pool_2.outputShape), pool_2.outputShape, conv3_filtersShape};

    ReLu3D relu3d_3{conv3.outputShape};

    xt::xarray<float> flatted = flatten(relu3d_3.output);
    int flattedSize = flatted.size();

    // ------------------------------------------------------------------------------

    Dense dense1{flattedSize, 2048};

    ReLu1D relu1D_1{dense1.outputShape};

    // ------------------------------------------------------------------------------

    Dense dense2{relu1D_1.outputShape, 2048};

    ReLu1D relu1D_2{dense2.outputShape};

    // ------------------------------------------------------------------------------

    Dense dense3{relu1D_2.outputShape, 1024};

    ReLu1D relu1D_3{dense3.outputShape};

    // ------------------------------------------------------------------------------

    Dense dense4{relu1D_2.outputShape, 512};

    ReLu1D relu1D_4{dense4.outputShape};

    // ------------------------------------------------------------------------------

    Dense dense5{relu1D_4.outputShape, 2};

    Softmax1D soft_1{dense5.outputShape};

    // ------------------------------------------------------------------------------

    conv1.forward(input) ;  // 32 x 44 x 44

    relu3d_1.forward(conv1.output);

    pool_1.forward(relu3d_1.output);    // 32 x 22 x 22

    // ------------------------------------------------------------------------------

    conv2.forward(pool_1.output);

    relu3d_2.forward(conv2.output);

    pool_2.forward(relu3d_2.output);

    // ------------------------------------------------------------------------------

    conv3.forward(pool_2.output);

    relu3d_3.forward(conv3.output);
    
    // ------------------------------------------------------------------------------

    dense1.dropout(50);

    dense1.forward(flatted);

    relu1D_1.forward(dense1.output);

    // ------------------------------------------------------------------------------

    dense2.dropout(50);

    dense2.forward(relu1D_1.output);

    relu1D_2.forward(dense2.output);

    // ------------------------------------------------------------------------------

    dense3.dropout(50);

    dense3.forward(relu1D_2.output);

    relu1D_3.forward(dense3.output);

    //std::cout << relu1D_3.output << std::endl;

    // ------------------------------------------------------------------------------

    dense4.dropout(50);

    dense4.forward(relu1D_3.output);

    // std::cout << dense4.output << std::endl;

    relu1D_4.forward(dense4.output);

    // ------------------------------------------------------------------------------

    dense5.dropout(50);

    dense5.forward(relu1D_4.output);

    soft_1.forward(dense5.output);

    return soft_1.output;
}