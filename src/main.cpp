#include <fstream>
#include <iostream>
#include <istream>
#include <string>
#include <tuple>
#include <xtensor/xarray.hpp>
#include <xtensor/xcsv.hpp>
#include <xtensor/xio.hpp>
#include <xtensor/xmanipulation.hpp>
#include <xtensor/xrandom.hpp>
#include <xtensor/xview.hpp>

#include "activation.h"
#include "const.h"
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

// 74%
NeuralNetwork CNN(std::tuple<int, int, int> inputShape, std::string name, float learningRate, LossType lossType, int batchSize, float validSplit, bool shuffle) {
	int inputShapeTotal = std::get<0>(inputShape) * std::get<1>(inputShape) * std::get<2>(inputShape);
	NeuralNetwork model = NeuralNetwork(name, learningRate, lossType, batchSize, validSplit, shuffle);

	Dense *dense1 = new Dense(inputShapeTotal, 32, relu, 0, false, true);
	Dense *dense2 = new Dense(32, 16, relu, 0, false, false);
	Dense *dense3 = new Dense(16, 10, relu, 0, false, false);

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
	// xt::random::seed(time(nullptr));
	xt::random::seed(42);

	NeuralNetwork nn = CNN(IMAGE_TENSOR_DIM, "topo3", 0.001, mse, 1, 0.0, true);

	// NeuralNetwork nn;
	// nn.load("../saves/topo1");

	std::vector<std::tuple<xt::xarray<float>, xt::xarray<float>>> trainSamples;
	std::vector<std::tuple<xt::xarray<float>, xt::xarray<float>>> testSamples;

	if (PNGPBM == 0) {
		trainSamples = loadingSets(trainPathPNG, nbImagesTrain);
		testSamples = loadingSets(evalPathPNG, nbImagesEval);
	} else {
		trainSamples = loadingSets(trainPathPBM, nbImagesTrain);
		testSamples = loadingSets(evalPathPBM, nbImagesEval);
	}

	nn.train(trainSamples, testSamples, 100, 10);

	std::cout << "Save ?" << std::endl;

	if (confirm()) {
		nn.save();
	}

	return 0;
}
