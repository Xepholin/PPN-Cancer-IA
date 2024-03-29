#include "dense.h"

#include <iostream>
#include <random>
#include <xtensor/xmath.hpp>
#include <mkl.h>

#include "tools.h"

void Dense::forward(xt::xarray<float> input) {
	if (this->flatten) {
		this->input = xt::flatten(input);
	} else {
		this->input = input;
	}

	this->dropout();

	cblas_sgemv(CblasRowMajor, CblasTrans, this->inputShape, this->outputShape, 1.0, this->weights.data(), this->outputShape, this->input.data(), 1, 0.0, this->output.data(), 1);	
	this->output += bias;

	this->baOutput = this->output;

	if (this->activation->name != "Activation") {
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

	vsMul(this->outputShape, gradient.data(), bnOutput.data(), bnOutput.data());
	vsAdd(this->outputShape, this->gammasGradient.data(), bnOutput.data(), this->gammasGradient.data());
	vsAdd(this->outputShape, gradient.data(), this->betasGradient.data(), this->betasGradient.data());


	// Calculer le gradient pour chaque neurone de sortie
	xt::xarray<float> layerGradient = xt::empty<float>({outputShape});
	xt::xarray<float> primeVector = xt::empty<float>({outputShape});

	for (int i = 0; i < outputShape; ++i)
	{
		primeVector(i) = this->activation->prime(baOutput(i));
	}

	vsMul(this->outputShape, this->gammas.data(), primeVector.data(), primeVector.data());
	vsMul(this->outputShape, primeVector.data(), gradient.data(), layerGradient.data());



	// Calculer les gradients des poids et des biais
	for (int i = 0; i < inputShape; ++i)
	{
		// Application du taux d'apprentissage déplacée ici
		cblas_saxpy(this->outputShape, this->input(i), layerGradient.data(), 1, this->weightsGradient.data() + i * outputShape, 1);
	}

	// Mise à jour des poids et des biais
	vsAdd(this->outputShape, layerGradient.data(), biasGradient.data(), biasGradient.data());

	// Accumulation correcte du gradient d'entrée
	xt::xarray<float> inputGradient = xt::empty<float>({inputShape});
	cblas_sgemv(CblasRowMajor, CblasNoTrans, this->inputShape, this->outputShape, 1.0f, this->weights.data(), this->outputShape, layerGradient.data(), 1, 0.0f, inputGradient.data(), 1);

	return inputGradient;
}


xt::xarray<float> Dense::oldbackward(
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

	float inputValue = 0.0;

    // Calculer les gradients des poids et des biais
    for (int i = 0; i < inputShape; ++i) {
		inputValue = input(i);

        for (int j = 0; j < outputShape; ++j) {
            this->weightsGradient(i, j) = this->weightsGradient(i, j) + (inputValue * layerGradient(j));
            // Application du taux d'apprentissage déplacée ici
        }
    }

    // Mise à jour des poids et des biais
    for (int i = 0; i < outputShape; ++i) {
        this->biasGradient(i) = this->biasGradient(i) + layerGradient(i);
    }

    xt::xarray<float> inputGradient = xt::empty<float>({inputShape});

    // // Accumulation correcte du gradient d'entrée
    // for (int i = 0; i < inputShape; ++i) {
    //     float sum = 0;
    //     for (int j = 0; j < outputShape; ++j) {
    //         sum += weights(i, j) * layerGradient(j);
    //     }
    //     inputGradient(i) = sum;
    // }

	for (int i = 0; i < inputShape; ++i)	{
		for (int j = 0; j < outputShape; ++j)	{
			inputGradient(i) = weights(i, j) * layerGradient(j);
		}
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
			this->input(i) = 0;
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