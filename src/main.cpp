#include <iostream>

#include "image.h"
#include "convolution.h"
#include "network.h"

#include "tools.h"

int main()
{
    int inputHeight = 3;
    int inputWidth = 3;
    int inputDepth = 3;

    int kernelSize = 2;

    Convolution conv(inputHeight, inputWidth, inputDepth, kernelSize, 64);  // Assuming RGB images with a 3x3 kernel and 64 filters

    std::cout << conv.input << std::endl;

    conv.forward(xt::xarray<float>({{1, 1, 1}, {1, 1, 1}}));
    conv.backward(xt::xarray<float>({{1, 1, 1}, {1, 1, 1}}));

    return 0;
}
