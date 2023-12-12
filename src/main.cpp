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

    NeuralNetwork nn;

    nn.add(new Dense{2304, 1024, relu});
    nn.add(new Dense{1024, 1024, relu});
    nn.add(new Dense{1024, 1024, relu});
    nn.add(new Dense{1024, 512, relu});
    nn.add(new Dense{512, 512, relu});
    nn.add(new Dense{512, 512, relu});
    nn.add(new Dense{512, 512, relu});
    nn.add(new Dense{512, 256, relu});
    nn.add(new Dense{256, 128, relu});
    nn.add(new Dense{128, 64, relu});
    nn.add(new Dense{64, 32, relu});
    nn.add(new Dense{32, 16, relu});
    nn.add(new Dense{16, 8, relu});
    nn.add(new Dense{8, 4, relu});
    nn.add(new Dense{4, 2, softmax});

    for (int i = 0; i < images.shape()[0]; ++i)  {
        xt::xarray<float> image = xt::view(images, i);
        nn.train(image, 1);
    }

    return 0;
}
