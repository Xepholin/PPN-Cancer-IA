#ifndef LAYER_H
#define LAYER_H

#include <tuple>
#include <string>

#include <xtensor/xarray.hpp>
#include <xtensor/xrandom.hpp>

// ILayer(xt::xarray<float> input, xt::xarray<float> output)
class ILayer
{

    public:
        std::string name = "ILayer";

        xt::xarray<float> input;
        xt::xarray<float> output;

        virtual void forward(xt::xarray<float> input);

        virtual void backward(xt::xarray<float> cost, float learningRate);

        virtual void print() const;
};

#endif