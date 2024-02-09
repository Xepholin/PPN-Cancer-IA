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
	// Create nn
    xt::random::seed(time(nullptr));

	NeuralNetwork nn = CNN2({1, 48, 48}, "test", 0.001, 0);

	// NeuralNetwork nn;
	// nn.load("../saves/test");

	// xt::xarray<float> image = importPBM("../../image/8863_idx5_x101_y1201_class0.pbm");

	// nn.iter(image, xt::xarray<float>{0, 1});

	// nn.load("../saves/toto");
	
    nn.train("../assets/processed/train", 1500);

    nn.eval("../assets/processed/eval");

	saveConfirm(nn, false);

	return 0;
}
