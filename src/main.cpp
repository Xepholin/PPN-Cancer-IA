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

	NeuralNetwork nn = CNN2({1, 48, 48}, "test", 0.0001, 0);

	// NeuralNetwork nn;
	// nn.load("../saves/test");

	// saveConfirm(nn, false);

	// nn.load("../saves/toto");
	
    nn.train("../assets/processed/train", 15);

    nn.eval("../assets/processed/eval");

	saveConfirm(nn, false);

	return 0;
}
