#include <iostream>
#include <tuple>
#include <string>
#include <xtensor/xarray.hpp>
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


#include <istream>
#include <fstream>
#include <iostream>
#include <xtensor/xcsv.hpp>


int main() {
	xt::random::seed(time(nullptr));

	std::vector<std::tuple<int, float>> resultTraining;
	xt::xarray<float> images = importAllPBM("../assets/PBM", 100);
	NeuralNetwork nn = CNN2(std::tuple{1, 48, 47}, 0.001, 50);

	resultTraining = nn.train(images, xt::xarray<float>{0, 1});

	saveConfirm(nn);

	return 0;
}
