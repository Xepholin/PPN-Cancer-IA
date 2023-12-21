#ifndef NETWORK_H
#define NETWORK_H

#include <vector>

#include "layer.h"
#include "dense.h"

#include <xtensor/xarray.hpp>
#include <xtensor/xrandom.hpp>

class NeuralNetwork 
{
    public:

        std::vector<ILayer*> nn;

        void add(ILayer *layer);

        void dropDense(uint16_t dropRate);

        void miniBatch(xt::xarray<float> batch, xt::xarray<int> trueLabels, uint16_t dropRate);

        void train(xt::xarray<float> input, xt::xarray<int> trueLabel);

        void detect( xt::xarray<float> input);

        void load(const char * path );

        void save(const char * path) const;
};

#endif