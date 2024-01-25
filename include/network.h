#ifndef NETWORK_H
#define NETWORK_H

#include <vector>

#include "layer.h"

#include <xtensor/xarray.hpp>
#include <xtensor/xrandom.hpp>

class NeuralNetwork 
{
    public:

        std::vector<ILayer*> nn;

        float learningRate;
        uint16_t dropRate;

        void add(ILayer *layer);

        void dropDense();

        void miniBatch(xt::xarray<float> batch, xt::xarray<int> label);

        void train(xt::xarray<float> input, xt::xarray<int> trueLabel);

        void detect(xt::xarray<float> input);

        void load(const std::string path);

        void save(const std::string path) const;

};

#endif