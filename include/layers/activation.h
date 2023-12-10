#ifndef ACTIVATION_H
#define ACTIVATION_H

#include <xtensor/xarray.hpp>
#include <xtensor/xio.hpp>
#include <xtensor/xrandom.hpp>

// Activation()
class Activation
{

    public:
        xt::xarray<float> input;
        xt::xarray<float> output;

        virtual void forward(xt::xarray<float> input);

        virtual void backward();
};

#endif