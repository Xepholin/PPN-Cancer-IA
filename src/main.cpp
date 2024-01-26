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
	xt::xarray<float> images = importAllPBM("../assets/PBM", 100);
	NeuralNetwork nn;
	
	// xt::xarray<int> label{1, 0};
	// xt::xarray<float> image = xt::empty<float>({1, 48, 48});
	// xt::view(image, 1) = xt::view(images, 0);
	
	// int i, j = 1;

	// while(i) {
	// 	nn.dropDense();
	// 	nn.train(image, label);
	// 	if (i % 2 == 0)	{
	// 		std::cout << nn.nn[nn.nn.size() - 1]->output << std::endl;
	// 		std::cout << MSE(nn.nn[nn.nn.size() - 1]->output, label) << std::endl;
	// 		j++;
	// 	}
	// 	else if (j % 2 == 0)	{
	// 		std::cout << nn.nn[nn.nn.size() - 1]->output << std::endl;
	// 		std::cout << MSE(nn.nn[nn.nn.size() - 1]->output, label) << std::endl;

	// 		int train = continueTraining();

	// 		if (!train)	{
	// 			break;
	// 		}
	// 		j++;
	// 	}

	// 	i++;
	// }

	// saveConfirm(nn);

	nn.load("../saves/marcleconnard");
	std::cout << nn.nn.size() <<std::endl;
	std::cout << nn.nn.size() <<std::endl;
	std::cout << nn.nn.size() <<std::endl;

	for (int i = 0; i < nn.nn.size(); ++i) {
		if (Convolution *conv= dynamic_cast<Convolution *>(nn.nn[i])) {

			std::cout << i << std::endl;
			std::cout << conv->filters << std::endl;
		}
	}

	return 0;
}
