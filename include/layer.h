#ifndef LAYER_H
#define LAYER_H

#include <tuple>

#include <xtensor/xarray.hpp>
#include <xtensor/xrandom.hpp>

// ILayer(xt::xarray<float> input, xt::xarray<float> output)
class ILayer
{

    public:
        xt::xarray<float> input;
        xt::xarray<float> output;

        virtual void forward(xt::xarray<float> input);

        virtual void backward(xt::xarray<float> gradient, float learningRate);

        virtual void print() const;
};

#endif