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


#include <istream>
#include <fstream>
#include <iostream>
#include <xtensor/xcsv.hpp>


int main() {
	xt::random::seed(time(nullptr));
	xt::xarray<float> images = importAllPBM("../assets/PBM", 1);
	NeuralNetwork nn = CNN2({1, 48, 48}, 0.001, 50);
	
	xt::xarray<int> label{0, 1};
	xt::xarray<float> image = xt::empty<float>({1, 48, 48});
	xt::view(image, 1) = xt::view(images, 0);

	for (int i = 0; i < 10; ++i) {
		nn.dropDense();
		nn.train(image, label);
		// if (i % 1000 == 0)	{
		// 	std::cout << nn.nn[nn.nn.size() - 1]->output << std::endl;
		// 	std::cout << MSE(nn.nn[nn.nn.size() - 1]->output, label) << std::endl;
		// }
	}

	nn.save("../saves/topologie_de_la_mort");

	return 0;
}
