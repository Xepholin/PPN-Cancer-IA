#ifndef NETWORK_H
#define NETWORK_H

#include <vector>
#include <xtensor/xarray.hpp>
#include <xtensor/xrandom.hpp>

#include "crossEntropy.h"
#include "layer.h"
#include "loss.h"
#include "mse.h"

class NeuralNetwork {
   public:
	std::string name;
	std::vector<ILayer *> nn;

	float learningRate;
	int nbEpoch = 0;
	float loss = 1000000.0;
	float accuracy = 0.0;
	int batchSize = 1;
	float validSplit = 0.2;
	bool shuffle = true;

	Loss *lossFunction;

	NeuralNetwork() = default;

	NeuralNetwork(std::string name, float learningRate = 0.1, LossType lossType = mse, int batchSize = 1, float validSplit = 0.2, bool shuffle = true) {
		this->name = name;
		this->learningRate = learningRate;
		this->batchSize = batchSize;
		this->validSplit = validSplit;
		this->shuffle = shuffle;

		switch (lossType) {
			case mse:
				this->lossFunction = new MSE();
				break;

			case cross_entropy:
				this->lossFunction = new CrossEntropy();
				break;

			default:
				break;
		}
	};

	~NeuralNetwork() = default;

	void add(ILayer *layer);

	void dropDense();

	void iter(xt::xarray<float> input, xt::xarray<int> trueLabel);

	void batch();

	void train(std::vector<std::tuple<xt::xarray<float>, xt::xarray<float>>> samples, std::vector<std::tuple<xt::xarray<float>, xt::xarray<float>>> validSamples, int epochs, int patience);

	void detect(xt::xarray<float> input);

	float eval(std::vector<std::tuple<xt::xarray<float>, xt::xarray<float>>> samples);

	void load(const std::string path);

	void save() const;
};

#endif