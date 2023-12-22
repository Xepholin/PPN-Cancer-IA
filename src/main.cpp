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

	xt::xarray<int> label{0, 1};
	xt::xarray<float> image = xt::empty<float>({1, 48, 48});
	xt::view(image, 1) = xt::view(images, 0);

	for (int i = 0; i < 100000; ++i) {
		nn.dropDense(50);
		nn.train(image, label);
		if ((i % 1000)==0) {
			std::cout << i << std::endl;
			std::cout << i << std::endl;
			std::cout << i << std::endl;
			std::cout << i << std::endl;
			std::cout << i << std::endl;
			std::cout << i << std::endl;
			std::cout << i << std::endl;
			std::cout << i << std::endl;
		}
	}

	// std::cout << conv1->filters << '\n' << std::endl;

	// std::cout << conv1->output << '\n' << std::endl;

	// Dense *conv = new Dense(4, 3, relu);
	// Dense *dense2 = new Dense(3, 2, softmax);

	// xt::xarray<float> input {{1.1, 1.2, 1.3, 1.4}};

	// xt::xarray<float> weight {{0.1, 0.2, 0.3}, {0.4, 0.5, 0.6}, {0.7, 0.8, 0.9}, {1, 1.1, 1.2}};
	// xt::xarray<float> weight2 {{0.1, 0.2}, {0.3, 0.4}, {0.5, 0.6}};
	// xt::xarray<float> label {1.0, 0.0};

	// dense->weights = weight;
	// dense2->weights = weight2;

	// dense->forward(input);
	// dense2->forward(dense->output);

	// std::cout << dense2->output << '\n' << std::endl;

	// float err = crossEntropy(dense2->output, label);

	// std::cout << err << '\n' << std::endl;

	// dense->backward(err, 0.01);

	return 0;
}
