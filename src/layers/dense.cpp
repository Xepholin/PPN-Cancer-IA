#include "dense.h"

#include <iostream>
#include <random>
#include <xtensor/xmath.hpp>

#include "tools.h"

void Dense::forward(xt::xarray<float> input) {
	if (this->flatten) {
		this->input = xt::flatten(input);
	} else {
		this->input = input;
	}

	this->dropout();

	this->output = dot_product_fma(this->weights,this->input) + bias;

	this->baOutput = this->output;

	if (this->activationType != ActivationType::ACTIVATION_NO_TYPE) {
		this->activation->forward(this->output);
		this->output = this->activation->output;
	}

	this->bnOutput = this->output;

	if (this->normalize)	{
		this->norm();
	}

	// std::cout << "Dense forward\n" << std::endl;
	// std::cout << "input:\n" << this->input << std::endl;
	// std::cout << "weights:\n" << this->weights << std::endl;
	// std::cout << "bias:\n" << this->bias << std::endl;
	// std::cout << "output:\n" << this->output << std::endl;
	
	// std::cout << std::endl;
	// std::cout << std::endl;
	// std::cout << std::endl;
	// std::cout << std::endl;
}

xt::xarray<float> Dense::backward(
    xt::xarray<float> gradient,
    float learningRate) 
{

	for (int i = 0; i < outputShape; ++i)	{
		this->gammasGradient(i) = this->gammasGradient(i) + (gradient(i) * bnOutput(i));
		this->betasGradient(i) = this->betasGradient(i) + gradient(i);
	}

	xt::xarray<float> layerGradient = xt::empty<float>({outputShape});

    // Calculer le gradient pour chaque neurone de sortie
    for (int i = 0; i < outputShape; ++i) {
        layerGradient(i) = this->activation->prime(baOutput(i)) * this->gammas(i) * gradient(i);
    }

    // Calculer les gradients des poids et des biais
    for (int i = 0; i < inputShape; ++i) {
        for (int j = 0; j < outputShape; ++j) {
            this->weightsGradient(j, i) = this->weightsGradient(j, i) + (input(i) * layerGradient(j));
            // Application du taux d'apprentissage déplacée ici
        }
    }

    // Mise à jour des poids et des biais
    for (int i = 0; i < outputShape; ++i) {
        this->biasGradient(i) = this->biasGradient(i) + layerGradient(i);
    }

    xt::xarray<float> inputGradient = xt::empty<float>({inputShape});

    // Accumulation correcte du gradient d'entrée
    for (int i = 0; i < inputShape; ++i) {
        float sum = 0;
        for (int j = 0; j < outputShape; ++j) {
            sum += weights(j, i) * layerGradient(j);
        }
        inputGradient(i) = sum;
    }

    return inputGradient;
}

void Dense::norm()	{
	auto mean = xt::mean(this->output);
	auto std = xt::stddev(this->output);

	this->output = (this->output - mean) / (std + 10e-6);

	for(int i = 0; i < outputShape; ++i)	{
		this->output(i) = (this->output(i) * this->gammas(i)) + this->betas(i);
	}
}

void Dense::print() const {
	std::cout << "Dense: " << this->output.shape()[0] << " fully connected neurons"
			  << "\n          |\n          v" << std::endl;
}

void Dense::printDropout(uint16_t dropRate) const {
	std::cout << "          | dropout p=" << dropRate << '%'
			  << "\n          v" << std::endl;
}

void Dense::dropout() {
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

	this->weights = xt::random::randn<float>({this->outputShape, this->inputShape}, 0, std);
}

void Dense::XGWeightsInit() {
	float std = sqrt(2.0 / (static_cast<float>(this->inputShape) + this->outputShape));

	this->weights = xt::random::randn<float>({this->outputShape, this->inputShape}, 0, std);
}