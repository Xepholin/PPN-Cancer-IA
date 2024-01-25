#include "network.h"

#include <xtensor/xio.hpp>
#include <xtensor/xview.hpp>

#include "dense.h"
#include "output.h"
#include "tools.h"

void NeuralNetwork::add(ILayer *layer) {
	this->nn.push_back(layer);
	return;
}

void NeuralNetwork::miniBatch(xt::xarray<float> batch, xt::xarray<int> trueLabels, uint16_t dropRate, float learningRate) {
	this->dropDense(dropRate);
	for (int i = 1; i < batch.shape()[0]; ++i) {
		this->train(xt::view(batch, i), trueLabels, learningRate);
	}
}

void NeuralNetwork::dropDense(uint16_t dropRate) {
	for (int i = 0; i < this->nn.size(); ++i) {
		if (Dense *dense = dynamic_cast<Dense *>(this->nn[i])) {
			dense->dropout(dropRate);
		}
	}
}

void NeuralNetwork::train(xt::xarray<float> input, xt::xarray<int> label, float learningRate) {
	this->nn[0]->forward(input);

	for (int i = 1; i < this->nn.size(); ++i) {
		this->nn[i]->forward(this->nn[i - 1]->output);
	}

	std::cout << "output: " << this->nn[this->nn.size() - 1]->output << std::endl;

	float error = MSE(this->nn[this->nn.size() - 1]->output, label);

	std::cout << "error: " << error << std::endl;

	xt::xarray<float> recycling;

	for (int i = this->nn.size() - 1; i >= 0; --i) {
		if (this->nn[i]->name == "Output") {
			recycling = this->nn[i]->backward(label, learningRate);
		}
		else if (this->nn[i]->name == "Dense") {
			recycling = this->nn[i]->backward(recycling, learningRate);
		}
		else	{
			break;
		}
	}
}

void NeuralNetwork::detect(xt::xarray<float> input) {}

void NeuralNetwork::load(const char *path) {}

void NeuralNetwork::save(const char *path) const {}