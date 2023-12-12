#include <iostream>
#include <random>

#include <xtensor/xio.hpp>

#include "dense.h"
#include "tools.h"

void Dense::forward(xt::xarray<float> input)
{
    this->input = input;

    for (int j = 0; j < this->weights.shape()[1]; ++j)
    {
        for (int i = 0; i < this->weights.shape()[0]; ++i)
        {
            if (drop(i) == true)
            {
                continue;
            }

            output(j) += this->weights(i, j) * this->input(i);
        }
    }

    this->output = batchNorm(this->output, this->beta, this->gamma);

    if (this->activationType != ActivationType::ACTIVATION_NO_TYPE) {
        this->activation->forward(this->output);
        this->output = this->activation->output;   
    }
}

void Dense::backward(
    float cost,
    float learningRate)
{
    xt::xarray<float> layerGradient = xt::empty<float>({this->weights.shape()[0]});

    // Calcul du gradient de l'erreur selon les sorties de la couche interne l
    for (int i = 0; i < this->weights.shape()[0]; ++i)
    {
        if(this->drop(i) == true)
        {
            continue;
        }

        float gradient = 0.0;
        for (int j = 0; j < this->weights.shape()[1]; ++j)
        {
            gradient += 2 * (cost - output(j)) * this->weights(j, i) * this->activation->prime(output(j));
        }
        layerGradient(i) = gradient;
    }

    // Calcul du gradient de l'erreur selon les poids pour la couche l
    xt::xarray<float> weightsGradient = xt::empty<float>({weights.shape()[0], weights.shape()[1]});
    for (int i = 0; i < weights.shape()[0]; ++i)
    {

        if(this->drop(i) == true){
            continue;
        }

        float gradient = 0.0;
        for (int j = 0; j < this->weights.shape()[1]; ++j)
        {

            weightsGradient(i, j) += layerGradient(j) * this->activation->prime(output(j));
        }
    }

    // Mise a jour des poids lie a la couche l
    for (int i = 0; i < weights.shape()[0]; ++i)
    {

        if(this->drop(i) == true){
            continue;
        }

        for (int j = 0; j < this->weights.shape()[1]; ++j)
        {   
            
            this->weights(i, j) -= learningRate * weightsGradient(i, j) * output(j);
        }
    }
}

void Dense::print() const
{
    std::cout << "Dense: " << this->output.shape()[0] << " fully connected neurons"
              << "\n          |\n          v" << std::endl;
}

void Dense::printDropout(uint16_t dropRate) const    {
    std::cout << "          | dropout p=" << dropRate << '%'
              << "\n          v" << std::endl;
}



void Dense::dropout(uint16_t dropRate)
{
    std::random_device rd;
    std::mt19937 gen(rd());


    for (int i = 0; i < this->weights.shape()[0]; ++i)
    {
        if (dropRate <= std::uniform_int_distribution<>(1, 100)(gen))
        {
            this->drop(i) = true;
        }
        else
        {
            this->drop(i) = false;
        }
    }
}