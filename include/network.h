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
    int nbEpoch = 0;
	float accuracy = 0.0;
	int batchSize = 1;

    NeuralNetwork() = default;

    NeuralNetwork(std::string name, float learningRate = 0.1)
    {
        this->name = name;
        this->learningRate = learningRate;
    };

    ~NeuralNetwork() = default;

    void add(ILayer *layer);

    void dropDense();

	void iter(xt::xarray<float> input, xt::xarray<int> trueLabel);

	void batch(int batchSize);

    std::vector<std::tuple<int, float>> train(const std::string path, int totalNumberImage, int batchSize);

    void detect(xt::xarray<float> input);

    void eval(const std::string path);

    void load(const std::string path);

    void save(const std::string path) const;
};

#endif