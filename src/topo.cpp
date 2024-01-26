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
#include "output.h"
#include "topo.h"

// NeuralNetwork CNN()
// {
//     NeuralNetwork nn;

// 	std::tuple<int, int, int> conv1_inputShape{1, 48, 48};
//     std::tuple<int, int, int, int, int> conv1_filtersShape{fil1, 6, 6, 2, 0};

//     // ------------------------------------------------------------------------------
//     Convolution* conv1 = new Convolution{1, conv1_inputShape, conv1_filtersShape, relu};
//     Pooling* pool_1 = new Pooling{conv1->outputShape, 2, 2, 0, PoolingType::POOLING_MAX};

//     // ------------------------------------------------------------------------------
//     // Changer le stride à 2 => 1024 flatted à 256
//     std::tuple<int, int, int, int, int> conv2_filtersShape{fil2, 5, 5, 1, 0};

//     // ------------------------------------------------------sys------------------------
//     Convolution* conv2 =  new Convolution{1, pool_1->outputShape, conv2_filtersShape, relu};
//     Pooling* pool_2 = new Pooling{conv2->outputShape, 2, 2, 0, PoolingType::POOLING_MAX};

//     // ------------------------------------------------------------------------------

//     xt::xarray<float> flatted = flatten(pool_2->output);
//     int flattedSize = flatted.size();

// }

NeuralNetwork CNN2(std::tuple<int, int, int> inputShape, std::string name, float learningRate, uint16_t dropRate)	{

    NeuralNetwork model;
	model.name = name;
	model.learningRate = learningRate;
    model.dropRate = dropRate;


    Convolution* conv1 = new Convolution{1, inputShape, std::tuple{1, 3, 3, 2, 0}, relu};

	// ------------------------------------------------------------------------------

    Pooling* pool_1 = new Pooling{conv1->outputShape, 2, 2, 0, PoolingType::POOLING_MAX};

    // ------------------------------------------------------------------------------

    Dense *dense1 = new Dense(pool_1->output.size(), 8, relu, false, true);

	// ------------------------------------------------------------------------------

    Output *output = new Output(dense1->output.size(), 2, softmax);

    // ------------------------------------------------------------------------------

    model.add(conv1);
    model.add(pool_1);
    model.add(dense1);
    model.add(output);

    return model;
}

NeuralNetwork CNN3(std::tuple<int, int, int> inputShape, std::string name, float learningRate, uint16_t dropRate){

    NeuralNetwork nn;
	nn.name = name;
    nn.learningRate = learningRate;
    nn.dropRate = dropRate;

    int fil1 = 6;
    int fil2 = 16;
    int fil3 = 16;

    // ------------------------------------------------------------------------------
	
    Convolution* conv1 = new Convolution{1, inputShape, std::tuple{fil1, 6, 6, 1, 0}, relu};
    Pooling* pool_1 = new Pooling{conv1->outputShape, 2, 2, 0, PoolingType::POOLING_MAX};

    // ------------------------------------------------------sys------------------------

    Convolution* conv2 =  new Convolution{1, pool_1->outputShape, std::tuple{fil2, 5, 5, 1, 0}, relu};
    Pooling* pool_2 = new Pooling{conv2->outputShape, 2, 2, 0, PoolingType::POOLING_MAX};

    // ------------------------------------------------------------------------------

    Dense *dense1 = new Dense(pool_2->output.size(), 84, relu, false, true);

    Dense *dense2 = new Dense(84, 32, relu);

	Output *output = new Output(32, 2, softmax);

    // ------------------------------------------------------------------------------

    nn.add(conv1);
    nn.add(pool_1);
    nn.add(conv2);
    nn.add(pool_2);
    nn.add(dense1);
    nn.add(dense2);
	nn.add(output);

    return nn;
}