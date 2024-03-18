#include <fstream>
#include <iostream>
#include <istream>
#include <string>
#include <tuple>
#include <xtensor/xarray.hpp>
#include <xtensor/xcsv.hpp>
#include <xtensor/xio.hpp>
#include <xtensor/xrandom.hpp>

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

NeuralNetwork CNN2(std::tuple<int, int, int> inputShape, std::string name, float learningRate, LossType lossType, int batchSize) {
	int inputShapeTotal = std::get<0>(inputShape) + std::get<1>(inputShape) + std::get<2>(inputShape); 
	NeuralNetwork model = NeuralNetwork(name, learningRate, lossType, batchSize);

	// ------------------------------------------------------------------------------

	// Convolution* conv1 = new Convolution{3, inputShape, std::tuple{16, 4, 4, 1, 0}, relu};
	// Pooling* pool_1 = new Pooling{conv1->outputShape, 3, 3, PoolingType::POOLING_MAX};

	// Convolution* conv2 = new Convolution{pool_1->depth, pool_1->outputShape, std::tuple{32, 4, 4, 1, 0}, relu};
	// Pooling* pool_2 = new Pooling{conv2->outputShape, 3, 3, PoolingType::POOLING_MAX};

	// Convolution* conv3 = new Convolution{pool_2->depth, pool_2->outputShape, std::tuple{64, 4, 4, 1, 0}, relu};
	// Pooling* pool_3 = new Pooling{conv3->outputShape, 3, 3, PoolingType::POOLING_MAX};

	// ------------------------------------------------------------------------------

	Dense *dense1 = new Dense(/*pool_1->output.size()*/ inputShapeTotal, 128, relu, 25, true, true);
	// Dense *dense2 = new Dense(dense1->outputShape, 128, relu, 25, true);
	// Dense *dense3 = new Dense(dense2->outputShape, 128, relu, 25, true);

	// ------------------------------------------------------------------------------

	Output *output = new Output(dense1->outputShape, 2, softmax);

	// ------------------------------------------------------------------------------

	// model.add(conv1);
	// model.add(pool_1);
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

int main() {
	xt::random::seed(time(nullptr));
	// xt::random::seed(42);

	NeuralNetwork nn = CNN2(IMAGE_TENSOR_DIM, "topo5", 0.0001, cross_entropy, 32);

	// NeuralNetwork nn;
	// nn.load("../saves/topo5");

	// std::cout << "nbEpoch: " << nn.nbEpoch << std::endl;

	// xt::xarray<float> image = importPBM("../../image/8863_idx5_x101_y1201_class0.pbm", 48);

	// nn.iter(image, xt::xarray<float>{0, 1});

	if (PNGPBM == 0)	{
		nn.train("../assets/breast/train");
		nn.eval("../assets/breast/eval");
	}
	else	{
		nn.train("../../processed1/train");
		nn.eval("../../processed1/eval");
	}

	saveConfirm(nn, false);

	// xt::xarray<float> images = xt::random::rand<float>({3, 5, 5});
	// xt::xarray<float> image = xt::empty<float>({5, 5});

	// xt::view(image, 0) = xt::view(images, 0);
	
	// std::cout << images << '\n' << std::endl;
	// std::cout << image << '\n' << std::endl;

	return 0;
}
