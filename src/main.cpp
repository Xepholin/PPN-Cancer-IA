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
    // xt::random::seed(time(nullptr));
    xt::xarray<float> images = importAllPBM("../assets/PBM", 3500);

    NeuralNetwork nn = CNN3();

    for (int i = 0; i < images.shape()[0]; ++i)  {
        xt::xarray<float> image = xt::empty<float>({1, 48, 48});
        xt::view(image, 1) = xt::view(images, i);
        nn.train(image, i&1);
    }

    // xt::xarray<float> input { 0.      ,  0.      ,  0.      ,  0.      ,  0.      ,  0.      , 
    //                         0.      ,  0.      ,  0.695926,  0.      ,  1.008194,  0.426802, 
    //                         0.725731,  0.      ,  0.      ,  0.848127,  0.134827,  0.      , 
    //                         0.      ,  0.      ,  0.309762,  0.      ,  0.836298,  0.380306, 
    //                         0.      ,  0.715518,  0.      ,  0.033389,  0.533112,  0.232409, 
    //                         0.612787,  0.      };

    // xt::xarray<float> weights  {{ 0.00001 , -0.00056 },
    //                         { 0.000319,  0.000424},
    //                         {-0.000036, -0.000164},
    //                         { 0.000417,  0.000005},
    //                         { 0.000043, -0.000229},
    //                         {-0.000456,  0.00036 },
    //                         {-0.000193, -0.000091},
    //                         { 0.000042,  0.000035},
    //                         { 0.000322,  0.000304},
    //                         {-0.000269, -0.00022 },
    //                         { 0.000226, -0.000014},
    //                         { 0.000102,  0.000394},
    //                         {-0.000011, -0.000042},
    //                         { 0.000126, -0.000651},
    //                         { 0.000051,  0.00015 },
    //                         {-0.000219, -0.000089},
    //                         { 0.000187,  0.000282},
    //                         {-0.000158, -0.000087},
    //                         {-0.000066,  0.00007 },
    //                         {-0.000049,  0.000113},
    //                         {-0.000038,  0.000109},
    //                         { 0.000321,  0.000316},
    //                         {-0.000221,  0.000045},
    //                         {-0.000167,  0.000086},
    //                         { 0.000084, -0.000215},
    //                         {-0.000092, -0.000313},
    //                         {-0.000145, -0.000166},
    //                         {-0.000311,  0.000207},
    //                         { 0.00024 ,  0.000084},
    //                         { 0.000083,  0.000353},
    //                         {-0.000142, -0.000609},
    //                         {-0.000057, -0.000143}};


    // xt::xarray<float> output = xt::zeros<float>({2});

    // for (int j = 0; j < weights.shape()[1]; ++j)
    // {
    //     for (int i = 0; i < weights.shape()[0]; ++i)
    //     {
    //         output(j) += weights(i, j) * input(i);
    //     }
    // }

    // std::cout << weights.shape()[0] << '\n' << std::endl;
    // std::cout << weights.shape()[1] << '\n' << std::endl;
    // std::cout << output << std::endl;


    // float expSum = xt::sum(xt::exp(output))();
    // float maxValue = xt::amax(output)();

    // for (int i = 0; i < output.shape()[0]; ++i)
    // {
    //     // auto exp_xi = std::exp(input(i) - maxValue);
    //     float exp_xi = std::exp(output(i));
    //     std::cout << "exp_xi " << exp_xi << std::endl;
    //     std::cout << "exp_sum " << expSum << std::endl;
    //     std::cout << output(i) << std::endl;
    //     output(i) = std::abs(exp_xi / expSum);
    //     std::cout << output(i) << std::endl;

    // }

    // std::cout << output << std::endl;

    return 0;
}
