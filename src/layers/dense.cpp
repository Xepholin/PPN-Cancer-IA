#include <iostream>
#include <random>

#include "dense.h"

float indicatrice(float x)
{
    if (x > 0)
    {
        return 1;
    }
    return 0;
}

void Dense::forward(xt::xarray<float> input)
{
    this->input = input;

    for (int i = 0; i < this->weights.shape()[0]; ++i)
    {
        for (int j = 0; j < this->weights.shape()[1]; ++j)
        {
            if (drop(i, j) & 0)
            {
                continue;
            }
            output(j) += this->weights(i, j) * this->input(i);
        }
    }

    std::cout << "Dense: " << this->output.shape()[0] << " fully connected neurons"
              << "\n          |\n          v" << std::endl;
}

void Dense::backward(
    xt::xarray<float> target,
    float tauxApprentissage)
{
    std::cout << "backward Dense" << std::endl;
    xt::xarray<float> layerGradient = xt::empty<float>({this->weights.shape()[0]});

    // Calcul du gradient de l'erreur selon les sorties de la couche interne l
    for (int i = 0; i < this->weights.shape()[0]; ++i)
    {
        float gradient = 0.0;
        for (int j = 0; j < this->weights.shape()[1]; ++j)
        {

            if(this->drop(i,j) == false){
                continue;
            }
            
            // Quel est la target pour une couche cachee ?
            gradient += 2 * (output(j) - target(j)) * this->weights(j, i) * indicatrice(output(j));
        }
        layerGradient(i) = gradient;
    }

    // Calcul du gradient de l'erreur selon les poids pour la couche l
    xt::xarray<float> weightsGradient = xt::empty<float>({weights.shape()[0], weights.shape()[1]});
    for (int i = 0; i < weights.shape()[0]; ++i)
    {
        float gradient = 0.0;
        for (int j = 0; j < weights.shape()[1]; ++j)
        {

            if(this->drop(i,j) == false){
                continue;
            }
            weightsGradient(i, j) += layerGradient(j) * indicatrice(output(j));
        }
    }

    // Mise a jour des poids lie a la couche l
    for (int i = 0; i < weights.shape()[0]; ++i)
    {
        for (int j = 0; j < weights.shape()[1]; ++j)
        {   
            
            if(this->drop(i,j) == false){
                continue;
            }
            this->weights(i, j) -= tauxApprentissage * weightsGradient(i, j) * output(j);
        }
    }
}



void Dense::dropout(uint8_t dropRate)
{
    std::random_device rd;
    std::mt19937 gen(rd());

    for (int i = 0; i < this->weights.shape()[0]; ++i)
    {
        for (int j = 0; j < this->weights.shape()[1]; ++j)
        {
            if (dropRate <= std::uniform_int_distribution<>(1, 100)(gen))
            {
                this->drop(i, j) = true;
            }
            else
            {
                this->drop(i, j) = false;
            }
        }
    }

    std::cout << "          | dropout p=" << dropRate
              << "\n          v" << std::endl;
}
