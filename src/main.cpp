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
#include "pooling.h"
#include "relu.h"
#include "softmax.h"
#include "tools.h"
#include "topo.h"

int main() {
	// xt::random::seed(time(nullptr));
	// xt::xarray<float> images = importAllPBM("../assets/PBM", 3500);
	// NeuralNetwork nn = CNN3();

	// for (int i = 0; i < 1000; ++i)  {
	// 	xt::xarray<int> label {0, 1};
	//     xt::xarray<float> image = xt::empty<float>({1, 48, 48});
	//     xt::view(image, 1) = xt::view(images, i);
	// 	nn.dropDense(50);
	//     nn.train(image, label);
	// 	nn.dropDense(0);
	// }

	NeuralNetwork nn;

	xt::xarray<float> input = {{{1, 2, 3, 4, 5, 6},
							   {7,8, 9, 10, 11, 12},
							   {13, 14, 15, 16, 17, 18},
							   {19, 20, 21, 22, 23, 24},
							   {25, 26, 27, 28, 29, 30},
							   {31, 32, 33, 34, 35, 36}}};


	std::tuple<int, int, int> conv1_inputShape{1, 6, 6};

	std::tuple<int, int, int, int, int> conv1_filtersShape{5, 2, 2, 1, 0};

	Convolution* conv1 = new Convolution{1, conv1_inputShape, conv1_filtersShape, relu};
	conv1->forward(input);


	Pooling* pool = new Pooling{conv1->outputShape, 2, 1, 0, POOLING_MAX};

	pool->forward(conv1->output);

	std::cout << pool->output << '\n' << std::endl; 

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
