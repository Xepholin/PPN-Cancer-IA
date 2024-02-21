#include <iostream>
#include <random>

#include "output.h"
#include "tools.h"

void Output::forward(xt::xarray<float> input)
{
	this->input = input;
	
	this->output = dot_product_fma(this->weights,this->input) + bias;

	// std::cout << output << std::endl;

	this->bOutput = this->output;

    if (this->normalize)
	{
        this->output = normalized(this->input);
    }

    if (this->activationType != ActivationType::ACTIVATION_NO_TYPE) {
        this->activation->forward(this->output);
        this->output = this->activation->output;
    }

	// std::cout << "Output forward\n" << std::endl;
	// std::cout << "input:\n" << this->input << std::endl;
	// std::cout << "weights:\n" << this->weights << std::endl;
	// std::cout << "bias:\n" << this->bias << std::endl;
	// std::cout << "output:\n" << this->output << std::endl;
	
	// std::cout << std::endl;
	// std::cout << std::endl;
	// std::cout << std::endl;
	// std::cout << std::endl;
}

xt::xarray<float> Output::backward(
	xt::xarray<float> label,
    float learningRate)
{
	xt::xarray<float> layerGradient = xt::empty<float>({outputShape});

	for (int i = 0; i < outputShape; ++i)	{
		layerGradient(i) = this->activation->prime(bOutput(i)) * (2.0 * (output(i) - label(i)));
	}

	xt::xarray<float> weightsGradient = xt::empty<float>({outputShape,inputShape});
	
	for (int i = 0; i < inputShape; ++i)	{
		for (int j = 0; j < outputShape; ++j)	{
			weightsGradient(j, i) = input(i) * layerGradient(j);
		}
	}


	xt::xarray<float> biasGradient = xt::empty<float>({outputShape});

	for (int i = 0; i < outputShape; ++i)	{
		biasGradient(i) = layerGradient(i);
	}

	weights = weights + (-learningRate) * weightsGradient;
	bias = bias + (-learningRate) * biasGradient;


	xt::xarray<float> inputGradient = xt::empty<float>({inputShape});

	for (int i = 0; i < inputShape; ++i)	{
		for (int j = 0; j < outputShape; ++j)	{
			inputGradient(i) = weights(j, i) * layerGradient(j);
		}
	}

	// std::cout << "Output backprop\n" << std::endl;
	// std::cout << "layerGradient:\n" << layerGradient << std::endl;
	// std::cout << "weightsGradient:\n" << weightsGradient << std::endl;
	// std::cout << "biasGradient:\n" << biasGradient << std::endl;
	// std::cout << "inputGradient:\n" << inputGradient << std::endl;
	// std::cout << std::endl;
	// std::cout << std::endl;
	// std::cout << std::endl;
	// std::cout << std::endl;

	return inputGradient;
}


void Output::print() const
{
    std::cout << "Output: " << this->output.shape()[0] << " fully connected neurons"
              << "\n          |\n          v" << std::endl;
}

void Output::heWeightsInit()    {
    float std = sqrt(2.0 / (static_cast<float>(this->inputShape)));

    this->weights = xt::random::randn<float>({this->inputShape, this->outputShape}, 0, std);
}

void Output::XGWeightsInit() {
    float std = sqrt(2.0 / (static_cast<float>(this->inputShape) + this->outputShape));

    this->weights = xt::random::randn<float>({this->inputShape, this->outputShape}, 0, std);
}