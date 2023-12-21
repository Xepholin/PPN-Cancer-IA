#include <iostream>

#include <tuple>
#include <vector>

#include <xtensor/xarray.hpp>
#include <xtensor/xrandom.hpp>
#include <xtensor/xio.hpp>

#include "convolution.h"
#include "activation.h"
#include "relu.h"
#include "softmax.h"
#include "pooling.h"
#include "dense.h"
#include "topo.h"

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

    Convolution conv1{1, conv1_inputShape, conv1_filtersShape, relu};

    Pooling pool_1{conv1.outputShape, 2, 2, 0, PoolingType::POOLING_MAX};

    // ------------------------------------------------------------------------------

    std::tuple<int, int, int, int, int> conv2_filtersShape{fil2, 3, 3, 1, 0};
    Convolution conv2{std::get<0>(pool_1.outputShape), pool_1.outputShape, conv2_filtersShape, relu};

    Pooling pool_2{conv2.outputShape, 2, 2, 0, PoolingType::POOLING_MAX};

    // ------------------------------------------------------------------------------

    std::tuple<int, int, int, int, int> conv3_filtersShape{fil2, 3, 3, 1, 0};
    Convolution conv3{std::get<0>(pool_2.outputShape), pool_2.outputShape, conv3_filtersShape, relu};

    xt::xarray<float> flatted = flatten(conv3.output);
    int flattedSize = flatted.size();

    // ------------------------------------------------------------------------------

    Dense dense1{flattedSize, 2048, relu};

    // ------------------------------------------------------------------------------

    Dense dense2{dense1.outputShape, 2048, relu};

    // ------------------------------------------------------------------------------

    Dense dense3{dense2.outputShape, 1024, relu};

    // ------------------------------------------------------------------------------

    Dense dense4{dense3.outputShape, 512, relu};

    // ------------------------------------------------------------------------------

    Dense dense5{dense4.outputShape, 2, softmax};

    // ------------------------------------------------------------------------------

    conv1.forward(input) ;  // 32 x 44 x 44

    pool_1.forward(conv1.output);    // 32 x 22 x 22

    // ------------------------------------------------------------------------------

    conv2.forward(pool_1.output);

    pool_2.forward(conv2.output);

    // ------------------------------------------------------------------------------

    conv3.forward(pool_2.output);
    
    // ------------------------------------------------------------------------------

    dense1.dropout(50);

    dense1.forward(flatted);

    // ------------------------------------------------------------------------------

    dense2.dropout(50);

    dense2.forward(dense1.output);

    // ------------------------------------------------------------------------------

    dense3.dropout(50);

    dense3.forward(dense2.output);

    // ------------------------------------------------------------------------------

    dense4.dropout(50);

    dense4.forward(dense3.output);

    // ------------------------------------------------------------------------------

    dense5.dropout(50);

    dense5.forward(dense4.output);

    return dense5.output;
}

xt::xarray<float> ANN(xt::xarray<float> input)
{

    Dense *dense1 = new Dense(100, 64, relu);

    Dense *dense2 = new Dense(64, 32, relu);

    Dense *dense3 = new Dense(32, 16, relu);

    Dense *dense4 = new Dense(16, 2, ActivationType::ACTIVATION_NO_TYPE);

    std::vector<ILayer *> topo = {dense1, dense2, dense3, dense4};

    topo[0]->forward(input);

    for (int i = 1; i < topo.size(); ++i)   {
        topo[i]->forward(topo[i-1]->output);
    }
    std::cout << topo[topo.size()-1]->output << std::endl;

    return topo[topo.size() - 1]->output;
}



NeuralNetwork CNN2(){

    NeuralNetwork nn;

    int fil1 = 6;
    int fil2 = 16;
    int fil3 = 16;

    std::tuple<int, int, int> conv1_inputShape{1, 48, 48};
    std::tuple<int, int, int, int, int> conv1_filtersShape{fil1, 6, 6, 2, 0};

    // ------------------------------------------------------------------------------
    Convolution* conv1 = new Convolution{1, conv1_inputShape, conv1_filtersShape, relu};
    Pooling* pool_1 = new Pooling{conv1->outputShape, 2, 2, 0, PoolingType::POOLING_MAX};

    // ------------------------------------------------------------------------------
    // Changer le stride à 2 => 1024 flatted à 256
    std::tuple<int, int, int, int, int> conv2_filtersShape{fil2, 5, 5, 1, 0};

    // ------------------------------------------------------sys------------------------
    Convolution* conv2 =  new Convolution{1, pool_1->outputShape, conv2_filtersShape, relu};
    Pooling* pool_2 = new Pooling{conv2->outputShape, 2, 2, 0, PoolingType::POOLING_MAX};

    // ------------------------------------------------------------------------------

    xt::xarray<float> flatted = flatten(pool_2->output);
    int flattedSize = flatted.size();

    // ------------------------------------------------------------------------------

    Dense *dense1 = new Dense(flattedSize, 128, relu, true);

    Dense *dense2 = new Dense(128, 96, relu);

    Dense *dense3 = new Dense(96, 64, relu);

    Dense *dense4 = new Dense(64, 45, relu);

    Dense *dense5 = new Dense(45, 24, relu);

    Dense *dense6 = new Dense(24, 16, relu);

    Dense *dense7 = new Dense(16, 8, relu);

    Dense *dense8 = new Dense(8, 2, softmax);

    // ------------------------------------------------------------------------------

    nn.add(conv1);
    nn.add(pool_1);
    nn.add(conv2);
    nn.add(pool_2);
    nn.add(dense1);
    nn.add(dense2);
    nn.add(dense3);
    nn.add(dense4);
    nn.add(dense5);
    nn.add(dense6);
    nn.add(dense7);
    nn.add(dense8);

    return nn;
}

NeuralNetwork CNN3(){

    NeuralNetwork nn;

    int fil1 = 6;
    int fil2 = 16;
    int fil3 = 16;

    std::tuple<int, int, int> conv1_inputShape{1, 48, 48};
    std::tuple<int, int, int, int, int> conv1_filtersShape{fil1, 6, 6, 2, 0};

    // ------------------------------------------------------------------------------
    Convolution* conv1 = new Convolution{1, conv1_inputShape, conv1_filtersShape, relu};
    Pooling* pool_1 = new Pooling{conv1->outputShape, 2, 2, 0, PoolingType::POOLING_MAX};

    // ------------------------------------------------------------------------------
    // Changer le stride à 2 => 1024 flatted à 256
    std::tuple<int, int, int, int, int> conv2_filtersShape{fil2, 5, 5, 1, 0};

    // ------------------------------------------------------sys------------------------
    Convolution* conv2 =  new Convolution{1, pool_1->outputShape, conv2_filtersShape, relu};
    Pooling* pool_2 = new Pooling{conv2->outputShape, 2, 2, 0, PoolingType::POOLING_MAX};

    // ------------------------------------------------------------------------------

    xt::xarray<float> flatted = flatten(pool_2->output);
    int flattedSize = flatted.size();

    // ------------------------------------------------------------------------------

    Dense *dense1 = new Dense(flattedSize, 84, relu);

    Dense *dense2 = new Dense(84, 32, relu);

    Dense *dense3 = new Dense(32, 2, softmax);

    // ------------------------------------------------------------------------------

    nn.add(conv1);
    nn.add(pool_1);
    nn.add(conv2);
    nn.add(pool_2);
    nn.add(dense1);
    nn.add(dense2);
    nn.add(dense3);

    return nn;
}