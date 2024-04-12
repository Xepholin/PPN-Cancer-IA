#include <fstream>
#include <iostream>
#include <istream>
#include <string>
#include <tuple>
#include <xtensor/xarray.hpp>
#include <xtensor/xcsv.hpp>
#include <xtensor/xio.hpp>
#include <xtensor/xview.hpp>
#include <xtensor/xrandom.hpp>
#include <xtensor/xmanipulation.hpp>

#include "activation.h"
#include "conv_op.h"
#include "convolution.h"
#include "dense.h"
#include "image.h"
#include "layer.h"
#include "network.h"
#include "output.h"
#include "pooling.h"
#include "relu.h"
#include "softmax.h"
#include "tools.h"
#include "const.h"

NeuralNetwork CNN2(std::tuple<int, int, int> inputShape, std::string name, float learningRate, LossType lossType, int batchSize, float validSplit, bool shuffle) {
	int inputShapeTotal = std::get<0>(inputShape) * std::get<1>(inputShape) * std::get<2>(inputShape); 
	NeuralNetwork model = NeuralNetwork(name, learningRate, lossType, batchSize, validSplit, shuffle);

	// ------------------------------------------------------------------------------

	Convolution* conv1 = new Convolution{inputShape, std::tuple{2, 4, 4, 1, 0}, relu};
	Pooling* pool_1 = new Pooling{conv1->outputShape, 3, 3, PoolingType::POOLING_MAX};

	// Convolution* conv2 = new Convolution{pool_1->depth, pool_1->outputShape, std::tuple{32, 4, 4, 1, 0}, relu};
	// Pooling* pool_2 = new Pooling{conv2->outputShape, 3, 3, PoolingType::POOLING_MAX};

	// Convolution* conv3 = new Convolution{pool_2->depth, pool_2->outputShape, std::tuple{64, 4, 4, 1, 0}, relu};
	// Pooling* pool_3 = new Pooling{conv3->outputShape, 3, 3, PoolingType::POOLING_MAX};

	// ------------------------------------------------------------------------------

	Dense *dense1 = new Dense(inputShapeTotal, 8, relu, 25, true, true);
	// Dense *dense2 = new Dense(dense1->outputShape, 256, relu, 25, true);
	// Dense *dense3 = new Dense(dense2->outputShape, 256, relu, 25, true);

	// ------------------------------------------------------------------------------

	Output *output = new Output(dense1->outputShape, 2, softmax);

	// ------------------------------------------------------------------------------

	model.add(conv1);
	model.add(pool_1);
	// model.add(conv2);
	// model.add(pool_2);
	// model.add(conv3);
	// model.add(pool_3);
	model.add(dense1);
	// model.add(dense2);
	// model.add(dense3);
	model.add(output);

	return model;
}

// 74%
NeuralNetwork CNN11(std::tuple<int, int, int> inputShape, std::string name, float learningRate, LossType lossType, int batchSize,float validSplit, bool shuffle)
{
	int inputShapeTotal = std::get<0>(inputShape) * std::get<1>(inputShape) * std::get<2>(inputShape); 
	NeuralNetwork model = NeuralNetwork(name, learningRate, lossType, batchSize);

	Dense *dense1 = new Dense(inputShapeTotal, 32, relu, 20, false, true);
	Dense *dense2 = new Dense(dense1->outputShape, 16, relu, 20, false, false);
	Dense *dense3 = new Dense(dense2->outputShape, 10, relu, 20, false, false);

	// ------------------------------------------------------------------------------

	Output *output = new Output(dense3->outputShape, 2, softmax);

	// ------------------------------------------------------------------------------

	model.add(dense1);
	model.add(dense2);
	model.add(dense3);
	model.add(output);

	return model;
}

int main() {
	xt::random::seed(time(nullptr));
	// xt::random::seed(42);

	NeuralNetwork nn = CNN11(IMAGE_TENSOR_DIM, "topo1", 0.001, mse, 1, 0.0, true);

	// NeuralNetwork nn;
	// nn.load("../saves/topo1");

	std::vector<std::tuple<xt::xarray<float>, xt::xarray<float>>> trainSamples;
	std::vector<std::tuple<xt::xarray<float>, xt::xarray<float>>> testSamples;

	if (PNGPBM == 0)	{
		trainSamples = loadingSets(trainPathPNG, nbImagesTrain);
		testSamples = loadingSets(evalPathPNG, nbImagesEval);
	}
	else	{
		trainSamples = loadingSets(trainPathPBM, nbImagesTrain);
		testSamples = loadingSets(evalPathPBM, nbImagesEval);
	}

	nn.train(trainSamples, testSamples, 100, 5, 0.2);

	// std::cout << "Save ?" << std::endl;

	// if (confirm())	{
	// 	nn.save();
	// }

	// xt::xarray<float> images = xt::random::rand<float>({5, 5, 5});
	
	// xt::xarray<float> split1 = xt::view(images, xt::range(0, 3));
	// xt::xarray<float> split2 = xt::view(images, xt::range(3, 5));

	// std::cout << images << '\n' << std::endl;
	// std::cout << split1 << '\n' << std::endl;
	// std::cout << split2 << '\n' << std::endl;

	return 0;
}
