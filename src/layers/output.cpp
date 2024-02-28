#include "output.h"

#include <cblas.h>

#include <iostream>
#include <random>
#include <xtensor/xmath.hpp>

#include "tools.h"

void Output::forward(xt::xarray<float> input) {
	this->input = input;

	this->dropout();
	
	cblas_sgemv(CblasRowMajor, CblasTrans, this->inputShape, this->outputShape, 1.0, this->weights.data(), this->outputShape, this->input.data(), 1, 0.0, this->output.data(), 1);	
	this->output += bias;

	this->baOutput = this->output;

	if (this->activationType != ActivationType::ACTIVATION_NO_TYPE) {
		this->activation->forward(this->output);
		this->output = this->activation->output;
	}

	this->bnOutput = this->output;

	if (this->normalize) {
		this->norm();
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
	float learningRate) {
	xt::xarray<float> normGradient = xt::empty<float>({outputShape});

	for (int i = 0; i < outputShape; ++i) {
		normGradient(i) = (2.0 * (output(i) - label(i)));

		this->gammasGradient(i) = this->gammasGradient(i) + (normGradient(i) * bnOutput(i));
		this->betasGradient(i) = this->betasGradient(i) + normGradient(i);
	}

	xt::xarray<float> layerGradient = xt::empty<float>({outputShape});

	// Calculer le gradient pour chaque neurone de sortie
	for (int i = 0; i < outputShape; ++i) {
		layerGradient(i) = this->activation->prime(baOutput(i)) * this->gammas(i) * normGradient(i);
	}

	// Calculer les gradients des poids et des biais
	for (int i = 0; i < inputShape; ++i) {
		for (int j = 0; j < outputShape; ++j) {
			this->weightsGradient(i, j) = this->weightsGradient(i, j) + (input(i) * layerGradient(j));
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
			sum += weights(i, j) * layerGradient(j);
		}
		inputGradient(i) = sum;
	}

	return inputGradient;
}

void Output::norm() {
	auto mean = xt::mean(this->output);
	auto std = xt::stddev(this->output);

	this->output = (this->output - mean) / (std + 10e-6);

	for (int i = 0; i < outputShape; ++i) {
		this->output(i) = (this->output(i) * this->gammas(i)) + this->betas(i);
	}
}

void Output::print() const {
	std::cout << "Output: " << this->output.shape()[0] << " fully connected neurons"
			  << "\n          |\n          v" << std::endl;
}

void Output::dropout() {
	std::random_device rd;
	std::mt19937 gen(rd());

	for (int i = 0; i < this->weights.shape()[0]; ++i) {
		if (dropRate >= std::uniform_int_distribution<>(1, 100)(gen)) {
			this->input(i) = 0;
		}
	}
}

void Output::heWeightsInit() {
	float std = sqrt(2.0 / (static_cast<float>(this->inputShape)));

	this->weights = xt::random::randn<float>({this->inputShape, this->outputShape}, 0, std);
}

void Output::XGWeightsInit() {
	float std = sqrt(2.0 / (static_cast<float>(this->inputShape) + this->outputShape));

	this->weights = xt::random::randn<float>({this->inputShape, this->outputShape}, 0, std);
}