#include "dense.h"

#include <iostream>
#include <random>

#include "tools.h"

void Dense::forward(xt::xarray<float> input) {
	if (this->flatten) {
		this->input = xt::flatten(input);
	} else {
		this->input = input;
	}

	// std::cout << "input\n" << this->input << '\n' << std::endl;
	// std::cout << "weights\n" << this->weights << '\n' << std::endl;

	for (int j = 0; j < this->outputShape; ++j) {
		float dotResult = 0;
		for (int i = 0; i < this->inputShape; ++i) {
			if (drop(i) == true) {
				continue;
			}

			dotResult += weights(i, j) * this->input(i);
		}

		this->output(j) = dotResult + bias(j);
	}

	this->bOutput = this->output;

	// std::cout << "before normalized output\n" << this->output << '\n' << std::endl;

	if (this->normalize) {
		this->output = normalized(this->output);
	}

	// std::cout << "before relu output\n" << this->output << '\n' << std::endl;

	if (this->activationType != ActivationType::ACTIVATION_NO_TYPE) {
		this->activation->forward(this->output);
		this->output = this->activation->output;
	}

}

xt::xarray<float> Dense::backward(
	xt::xarray<float> gradient,
	float learningRate) 
{
	xt::xarray<float> layerGradient = xt::empty<float>({outputShape});

	for (int i = 0; i < outputShape; ++i)	{
		layerGradient(i) = this->activation->prime(bOutput(i)) * (2.0 * (output(i) - gradient(i)));
	}

	xt::xarray<float> weightsGradient = xt::empty<float>({inputShape, outputShape});
	
	for (int i = 0; i < inputShape; ++i)	{
		for (int j = 0; j < outputShape; ++j)	{
			weightsGradient(i, j) = input(i) * layerGradient(j);
			weightsGradient(i, j) = (-learningRate) * weightsGradient(i, j);
		}
	}

	xt::xarray<float> biasGradient = xt::empty<float>({outputShape});

	for (int i = 0; i < outputShape; ++i)	{
		biasGradient(i) = layerGradient(i);
		biasGradient(i) = (-learningRate) * biasGradient(i);
	}

	weights = weights + weightsGradient;
	bias = bias + biasGradient;

	xt::xarray<float> inputGradient = xt::empty<float>({inputShape});
	
	for (int i = 0; i < inputShape; ++i)	{
		for (int j = 0; j < outputShape; ++j)	{
			inputGradient(i) = weights(i, j) * layerGradient(j);
		}
	}

	return inputGradient;
}

void Dense::print() const {
	std::cout << "Dense: " << this->output.shape()[0] << " fully connected neurons"
			  << "\n          |\n          v" << std::endl;
}

void Dense::printDropout(uint16_t dropRate) const {
	std::cout << "          | dropout p=" << dropRate << '%'
			  << "\n          v" << std::endl;
}

void Dense::dropout(uint16_t dropRate) {
	std::random_device rd;
	std::mt19937 gen(rd());

	for (int i = 0; i < this->weights.shape()[0]; ++i) {
		if (dropRate >= std::uniform_int_distribution<>(1, 100)(gen)) {
			this->drop(i) = true;
		} else {
			this->drop(i) = false;
		}
	}
}

void Dense::heWeightsInit() {
	float std = sqrt(2.0 / (static_cast<float>(this->inputShape)));

	this->weights = xt::random::randn<float>({this->inputShape, this->outputShape}, 0, std);
}

void Dense::XGWeightsInit() {
	float std = sqrt(2.0 / (static_cast<float>(this->inputShape) + this->outputShape));

	this->weights = xt::random::randn<float>({this->inputShape, this->outputShape}, 0, std);
}