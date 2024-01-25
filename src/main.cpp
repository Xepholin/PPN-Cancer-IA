#include <iostream>
#include <tuple>
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

int main() {
	xt::random::seed(time(nullptr));
	xt::xarray<float> images = importAllPBM("../assets/PBM", 1);
	NeuralNetwork nn = CNN3({1, 48, 48});

	xt::xarray<int> label{1, 0};
	xt::xarray<float> image = xt::empty<float>({1, 48, 48});
	xt::view(image, 1) = xt::view(images, 0);

	for (int i = 0; i < 100000; ++i) {
		// nn.dropDense(50);
		nn.train(image, label, 0.001);
	}
	
	return 0;
}
