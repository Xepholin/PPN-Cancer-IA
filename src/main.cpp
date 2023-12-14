#include <iostream>
#include <tuple>

#include <xtensor/xarray.hpp>
#include <xtensor/xio.hpp>
#include <xtensor/xrandom.hpp>

#include "image.h"
#include "tools.h"
#include "conv_op.h"
#include "layer.h"

#include "convolution.h"
#include "activation.h"
#include "relu.h"
#include "softmax.h"
#include "pooling.h"
#include "dense.h"
#include "topo.h"

#include "network.h"

int main()
{   
    xt::xarray<float> images = importAllPBM("../assets/PBM", 3500);

    NeuralNetwork nn = CNN2();

    for (int i = 0; i < images.shape()[0]; ++i)  {
        xt::xarray<float> image = xt::empty<float>({1, 48, 48});
        xt::view(image, 1) = xt::view(images, i);
        nn.train(image, i&1);
        break;
    }

    return 0;
}
