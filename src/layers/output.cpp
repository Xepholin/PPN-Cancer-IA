#include <iostream>
#include <random>

#include "output.h"
#include "tools.h"

void Output::forward(xt::xarray<float> input)
{   
    if (this->flatten)  {
        this->input = xt::flatten(input);
    }
    else
	{
        this->input = input;
    }

	for (int j = 0; j < this->outputShape; ++j)
    {
        float dotResult = 0;
        for (int i = 0; i < this->inputShape; ++i)
        {
            if (drop(i) == true)
            {
                continue;
            }

            dotResult += weights(i, j) * this->input(i);
        }
        this->output(j) = dotResult;
    }

    if (this->normalize)
	{
        this->output = instNorm(this->input);
    }

	this->bOutput = this->output;

    if (this->activationType != ActivationType::ACTIVATION_NO_TYPE) {
        this->activation->forward(this->output);
        this->output = this->activation->output;
    }
    
}

void Output::backward(
    float cost,
    float learningRate,
    xt::xarray<int> trueLabel)
{
    xt::xarray<float> layerGradientOutput = xt::empty<float> ({this->output.shape()[0]});

    // Calcul du gradient de cross entropy par rapport à l'output de Softmax
    for (int i = 0; i < this->output.shape()[0]; ++i)
    {
        layerGradientOutput(i) = this->output(i) - trueLabel(i);
    }

    xt::xarray<float> layerGradient = xt::empty<float>({this->weights.shape()[1]});

	// Calcul du gradient de la MSE selon les sorties de la couche d'output
	for (int i = 0; i < this->weights.shape()[1]; ++i) {
		float gradient = 0.0;
		for (int j = 0; j < this->weights.shape()[0]; ++j) {

			gradient +=  layerGradientOutput(j)* this->weights(j, i) * this->activation->prime(bOutput(j));
		}
		exit(0);
		layerGradient(i) = gradient;
	}

	std::cout << "layerGradient\n"
			  << layerGradient << '\n'
			  << std::endl;

	// Calcul du gradient de l'erreur selon les poids pour la couche de sortie
	xt::xarray<float> weightsGradient = xt::empty<float>({weights.shape()[0], weights.shape()[1]});
	for (int i = 0; i < weights.shape()[0]; ++i) {
		if (this->drop(i) == true) {
			continue;
		}

		float gradient = 0.0;
		for (int j = 0; j < this->weights.shape()[1]; ++j) {
			weightsGradient(i, j) = layerGradient(j) * this->activation->prime(bOutput(j));
		}
	}

	// Mise a jour des poids lie a la couche de sortie
	for (int i = 0; i < weights.shape()[0]; ++i) {
		if (this->drop(i) == true) {
			continue;
		}

		for (int j = 0; j < this->weights.shape()[1]; ++j) {
			this->weights(i, j) -= learningRate * weightsGradient(i, j) * bOutput(j);
		}
	}
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