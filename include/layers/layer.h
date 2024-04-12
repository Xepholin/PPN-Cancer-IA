#ifndef LAYER_H
#define LAYER_H

#include <tuple>
#include <string>

#include <xtensor/xarray.hpp>
#include <xtensor/xrandom.hpp>
#include <xtensor/xview.hpp>

#include <xtensor/xio.hpp>

class ILayer
{

    public:
        std::string name = "ILayer";

        xt::xarray<float> input;
        xt::xarray<float> output;

        virtual void forward(xt::xarray<float> input, bool training = true);

        virtual xt::xarray<float> backward(
				xt::xarray<float> gradient,
    			float learningRate);

        virtual void print() const;
};

#endif