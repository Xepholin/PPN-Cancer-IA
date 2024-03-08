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

NeuralNetwork CNN2(std::tuple<int, int, int> inputShape, std::string name, float learningRate, LossType lossType) {
	int inputShapeTotal = std::get<0>(inputShape) + std::get<1>(inputShape) + std::get<2>(inputShape); 
	NeuralNetwork model = NeuralNetwork(name, learningRate, lossType);

	// ------------------------------------------------------------------------------

	// Convolution* conv1 = new Convolution{1, inputShape, std::tuple{3, 3, 3, 1, 0}, relu};
	// Pooling* pool_1 = new Pooling{conv1->outputShape, 2, 2, PoolingType::POOLING_MAX};

	// ------------------------------------------------------------------------------

	Dense *dense1 = new Dense(/*pool_1->output.size()*/ inputShapeTotal, 64, relu, 25, true, true);

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
	xt::random::seed(time(nullptr));
	// xt::random::seed(42);

	NeuralNetwork nn = CNN2(IMAGE_TENSOR_DIM, "topo3", 0.0001, cross_entropy);

	// NeuralNetwork nn;
	// nn.load("../saves/topo2");

	// std::cout << "nbEpoch: " << nn.nbEpoch << std::endl;

	// xt::xarray<float> image = importPBM("../../image/8863_idx5_x101_y1201_class0.pbm", 48);

	// nn.iter(image, xt::xarray<float>{0, 1});

	if (PNGPBM == 0)	{
		nn.train("../assets/breast/train", 1);

		nn.eval("../assets/breast/eval");
	}
	else	{
		nn.train("../../processed1/train", 1);

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
