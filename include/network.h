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
	float accuracy = 0.0;
	int batchSize = 1;

	Loss *lossFunction;

	NeuralNetwork() = default;

	NeuralNetwork(std::string name, float learningRate = 0.1, LossType lossType = mse) {
		this->name = name;
		this->learningRate = learningRate;

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

	void batch(int batchSize);

	std::vector<std::tuple<int, float>> train(const std::string path, int batchSize);

	void detect(xt::xarray<float> input);

	void eval(const std::string path);

	void load(const std::string path);

	void save(const std::string path) const;
};

#endif