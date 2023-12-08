#include <iostream>
#include <tuple>

#include "image.h"
#include "convolution.h"
#include "network.h"
#include "activations.h"

#include "tools.h"




int main()
{


    xt::xarray<float> input = xt::random::randint({3,3,3}, 0, 10);
    auto input2 = input;
    // xt::xarray<float> input = xt::random::randint({2,3, 5, 5}, -10, 10);  
    std::cout << input << '\n' << std::endl;
    // std::cout << xt::where(input <= 0, 0.0, input) << std::endl;
    
    std::cout << input << '\n' << std::endl;



    return 0;
}



void CNN(){

    int fil1 = 32;
    int fil2 = 64;
    int fil3 = 128;
    
    xt::xarray<float> input = xt::random::randint({1,48, 58},0,2);  

    // ------------------------------------------------------------------------------
    std::tuple<int, int, int> conv1_inputShape{1, 48, 48};
    std::tuple<int, int, int, int, int> conv1_filtersShape{fil1, 5, 5, 1, 0};
    ConvolutionLayer conv1{1, conv1_inputShape, conv1_filtersShape};

    conv1.forward(input) ;  // 32 x 44 x 44


    std::cout << conv1.output << std::endl;


    ReLu3D relu3d_1{conv1.outputShape};
    relu3d_1.forward(conv1.output);

    std::cout << relu3d_1.output << std::endl;

    PoolingLayer pool_1{relu3d_1.outputShape,2,2,0,PoolingLayer::MAX};
    pool_1.forward(relu3d_1.output);    // 32 x 22 x 22



    // ------------------------------------------------------------------------------

    std::tuple<int, int, int, int, int> conv2_filtersShape{fil2, 3, 3, 1, 0};
    ConvolutionLayer conv2{std::get<0>(pool_1.outputShape), pool_1.outputShape, conv2_filtersShape};

    ReLu3D relu3d_2{conv2.outputShape};
    relu3d_2.forward(conv2.output);

    PoolingLayer pool_2{relu3d_2.outputShape,2,2,0,PoolingLayer::MAX};
    pool_2.forward(relu3d_2.output);

    // ------------------------------------------------------------------------------

    std::tuple<int, int, int, int, int> conv3_filtersShape{fil2, 3, 3, 1, 0};
    ConvolutionLayer conv3{std::get<0>(pool_2.outputShape), pool_2.outputShape, conv3_filtersShape};

    ReLu3D relu3d_3{conv3.outputShape};
    relu3d_3.forward(conv3.output);


    // ------------------------------------------------------------------------------

    xt::xarray<float> flatted = flatten(relu3d_3.output);

    // ------------------------------------------------------------------------------

    DenseLayer dense1{4096, 2048};

    dense1.forward(flatted);

    ReLu1D relu1D1{dense1.inputShape, dense1.outputShape};

    relu1D1.forward(dense1.output);

    dense1.dropout(50);

    // ------------------------------------------------------------------------------

    DenseLayer dense2{2048, 1048};

    dense2.forward(relu1D1.output);

    ReLu1D relu1D2{dense2.inputShape, dense2.outputShape};

    relu1D2.forward(dense2.output);

    dense2.dropout(50);

    // ------------------------------------------------------------------------------

    DenseLayer dense3{1048, 2};

    dense3.forward(relu1D2.output);

    // ------------------------------------------------------------------------------

    std::cout << "Prédiction" << '\n' << "---------------" << std::endl;

    std::cout << dense3.output << std::endl;
}
