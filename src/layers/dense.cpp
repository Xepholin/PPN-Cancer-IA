#include "dense.h"

#include <iostream>
#include <random>

#include "tools.h"

void Dense::forward(xt::xarray<float> input) {

	std::cout << "Dense\n" << std::endl;

	if (this->flatten) {
		this->input = xt::flatten(input);
	} else {
		this->input = input;
	}

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

	if (this->normalize) {
		this->output = normalized(this->output);
	}

	if (this->activationType != ActivationType::ACTIVATION_NO_TYPE) {
		this->activation->forward(this->output);
		this->output = this->activation->output;
	}

	std::cout << "input:\n" << this->input << std::endl;
	std::cout << "weights:\n" << this->weights << std::endl;
	std::cout << "bias:\n" << this->bias << std::endl;
	std::cout << "output:\n" << this->output << std::endl;
	
	std::cout << std::endl;
	std::cout << std::endl;
	std::cout << std::endl;
	std::cout << std::endl;
}

xt::xarray<float> Dense::backward(
	xt::xarray<float> gradient,
	float learningRate) 
{
	std::cout << "Dense\n" << std::endl;
	
	xt::xarray<float> layerGradient = xt::empty<float>({outputShape});

	for (int i = 0; i < outputShape; ++i)	{
		layerGradient(i) = this->activation->prime(bOutput(i)) *  gradient(i);
	}

	xt::xarray<float> weightsGradient = xt::empty<float>({inputShape, outputShape});
	
	for (int i = 0; i < inputShape; ++i)	{
		for (int j = 0; j < outputShape; ++j)	{
			weightsGradient(i, j) = input(i) * layerGradient(j);
		}
	}

	xt::xarray<float> biasGradient = xt::empty<float>({outputShape});

	for (int i = 0; i < outputShape; ++i)	{
		biasGradient(i) = layerGradient(i);
	}

	std::cout << "weights:\n" << this->weights << std::endl;

	weights = weights + (-learningRate) * weightsGradient;
	bias = bias + (-learningRate) * biasGradient;

	xt::xarray<float> inputGradient = xt::empty<float>({inputShape});
	
	for (int i = 0; i < inputShape; ++i)	{
		for (int j = 0; j < outputShape; ++j)	{
			inputGradient(i) = weights(i, j) * layerGradient(j);
		}
	}

	std::cout << "layerGradient:\n" << layerGradient << std::endl;
	std::cout << "weightsGradient:\n" << weightsGradient << std::endl;
	std::cout << "biasGradient:\n" << biasGradient << std::endl;
	std::cout << "inputGradient:\n" << inputGradient << std::endl;
	std::cout << std::endl;
	std::cout << std::endl;
	std::cout << std::endl;
	std::cout << std::endl;

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