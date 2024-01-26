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
#include "topo.h"

int main() {
	generateAllPBM("../assets/breast", "../assets/PBM");

	// xt::random::seed(time(nullptr));

	// std::vector<std::tuple<int, float>> resultTraining;

	// xt::xarray<float> images = importAllPBM("../assets/PBM/eval/0", 3500);

	// bool loaded = false;

	// NeuralNetwork nn = CNN2({1, 48, 48}, "toto", 0.001, 50);

	// resultTraining = nn.train(images, xt::xarray<float>{0, 1});

	// saveConfirm(nn, loaded);

	return 0;
}
