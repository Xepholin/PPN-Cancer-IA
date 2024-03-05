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

NeuralNetwork CNN2(std::tuple<int, int, int> inputShape, std::string name, float learningRate, LossType lossType) {
	NeuralNetwork model = NeuralNetwork(name, learningRate, lossType);

	// ------------------------------------------------------------------------------

	// Convolution* conv1 = new Convolution{1, inputShape, std::tuple{3, 3, 3, 1, 0}, relu};
	// Pooling* pool_1 = new Pooling{conv1->outputShape, 2, 2, PoolingType::POOLING_MAX};

	// ------------------------------------------------------------------------------

	Dense *dense1 = new Dense(/*pool_1->output.size()*/ 48*48, 64, relu, 25, true, true);

	// ------------------------------------------------------------------------------

	Output *output = new Output(dense1->outputShape, 2, softmax);

	// ------------------------------------------------------------------------------

	// model.add(conv1);
	// model.add(pool_1);
	model.add(dense1);
	model.add(output);

	return model;
}

int main() {
	// Create nn
	// xt::random::seed(time(nullptr));
	xt::random::seed(42);

	NeuralNetwork nn = CNN2({1, 48, 48}, "topo3", 0.0001, cross_entropy);

	// NeuralNetwork nn;
	// nn.load("../saves/topo2");

	// std::cout << "nbEpoch: " << nn.nbEpoch << std::endl;

	// xt::xarray<float> image = importPBM("../../image/8863_idx5_x101_y1201_class0.pbm");

	// nn.iter(image, xt::xarray<float>{0, 1});

	nn.train("../../processed/train", 150000, 16);

	nn.eval("../../processed/eval");

	saveConfirm(nn, false);

	// xt::random::seed(42);

	// Convolution* conv1 = new Convolution{1, {1, 5, 5}, std::tuple{1, 2, 2, 1, 0}, relu};
	// xt::xarray<float> image = xt::random::rand<float>({1, 5, 5});

	// conv1->forward(image);

	// std::cout << conv1->output << std::endl;

	return 0;
}
