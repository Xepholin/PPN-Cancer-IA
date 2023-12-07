#include "activations.h"
#include <iostream>

#include <xtensor/xarray.hpp>
#include <xtensor/xio.hpp>
#include <xtensor/xrandom.hpp>

void ReLu::forward(xt::xarray<float> input)
{
    this->input = input;
    std::cout << "ReLu forward" << std::endl;
    
    for (std::size_t i = 0; i < input.shape()[0]; ++i)
    {
        // Apply the activation function to the 2D slice
        auto activation_result = this->activation(xt::view(input, i));
        xt::view(output, i) = activation_result;
    }
}

float indicatrice (float x)
{
    if (x > 0)
    {
        return 1;
    }

    return 0;
}


void ReLu::backward(xt::xarray<float> gradient)
{
    std::cout << "ReLu backward" << std::endl;
}

xt::xarray<float> ReLu::activation(xt::xarray<float> matrix)
{
    return xt::where(matrix <= 0, 0.0, matrix);
}

// profondeur indice  x y


//void backward(xt::xarray<float> gradient,
//                xt::xarray<float> y,
//                xt::xarray<float> aL,
//                xt::xarray<float> z,
//                xt::xarray<float> zL, 
//                xt::xarray<float> aNext,
//                xt::xarray<float> wNext)
//{
//    float delta = .1;


    // aL : output de ReLU à la dernière couche du CNN
    // aBefore : output de ReLU à la couche L-1
//    xt::xarray<float> errL = xt::empty<float>({1, 2});
//    errL (1) = 2*(aL(1) - y(1))*aNext(i)*indicatrice(z(1));
//    errL (2) = 2*(aL(2) - y(2))*aNext(i)*indicatrice(z(2)); 

//    for (int i)
//    {
//        for (int j)
//        {
//            err = indicatrice(z(i))*aL(i)*2*(a(j) - y(j)*wNext(j, i)*indicatrice(zNext(j))) ;
//            w(j, i) += delta*err(i);
//        }
//    }
    

//}