#ifndef NETWORK_H
#define NETWORK_H

#include <vector>

#include "layer.h"

#include <xtensor/xarray.hpp>
#include <xtensor/xrandom.hpp>

class NeuralNetwork
{
public:
    std::string name;
    std::vector<ILayer *> nn;

    float learningRate;
    uint16_t dropRate;
    int nbEpoch = 1;

    NeuralNetwork() = default;

    NeuralNetwork(std::string name, float learningRate = 0.1, uint16_t dropRate = 50)
    {
        this->name = name;
        this->learningRate = learningRate;
        this->dropRate = dropRate;
    };

    ~NeuralNetwork() = default;

    void add(ILayer *layer);

    void dropDense();

    void miniBatch(xt::xarray<float> batch, xt::xarray<int> label);

    void iter(xt::xarray<float> input, xt::xarray<int> trueLabel);

    std::vector<std::tuple<int, float>> train(xt::xarray<float> dataset, xt::xarray<int> label);

    void detect(xt::xarray<float> input);

    void eval(const std::string path);

    void load(const std::string path);

    void save(const std::string path) const;
};

#endif