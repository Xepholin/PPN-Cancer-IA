#include <iostream>
#include <random>

#include "dense.h"

void Dense::forward(xt::xarray<float> input)
{
    this->input = input;

    for (int i = 0; i < this->weights.shape()[0]; ++i)
    {
        for (int j = 0; j < this->weights.shape()[1]; ++j)
        {
            output(j) += this->weights(i, j) * this->input(i);
        }
    }

    std::cout << "Dense: " <<
    this->output.shape()[0] << " fully connected neurons" <<
    "\n          |\n          v" << std::endl;
}

float indicatrice(float x)
{
    if (x > 0)
    {
        return 1;
    }
    return 0;
}

float lossFunction(xt::xarray<float> outputL,
                   xt::xarray<float> target)
{
    float err = 0.0;
    for (int i = 0; i < outputL.size(); ++i)
    {
        err += 0.5 * ((outputL(i) - target(i)) * (outputL(i) - target(i)));
    }

    return err;
}

void Dense::backward(
    xt::xarray<float> target,
    xt::xarray<float> outputL,
    xt::xarray<float> weightsL,
    float tauxApprentissage)
{
    std::cout << "backward Dense" << std::endl;

    // Calcul du gradient de l'erreur selon le poids pour la couche L
    xt::xarray<float> errL;
    for (int i = 0; i < outputL.size(); ++i)
    {
        float gradient = 0.0;
        // ligne de la matrice poids ?
        for (int j = 0; j < weightsL.shape()[0]; ++j)
        {
            gradient += 2 * (outputL(i) - target(i)) * indicatrice(outputL(i));
        }
        // errL.push_back(gradient);
    }

    // Mise a jour des poids lie a la couche de sortie
    for (int i = 0; i < weightsL.shape()[0]; ++i)
    {
        for (int j = 0; j < weightsL.shape()[1]; ++j)
        {
            weightsL(i, j) -= tauxApprentissage * errL(j) * outputL(j);
        }
    }
}

void Dense::backwardHiddenLayer(
    xt::xarray<float> target,
    xt::xarray<float> output,
    xt::xarray<float> weights,
    float tauxApprentissage)
{
    std::cout << "backward Dense" << std::endl;
    // Calcul du gradient de l'erreur selon les sorties de la couche interne l
    xt::xarray<float> layerGradient;
    for (int i = 0; i < weights.shape()[0]; ++i)
    {
        float gradient = 0.0;
        for (int j = 0; i < weights.shape()[1]; ++j)
        {
            // Quel est la target pour une couche cachee ?
            // flatten output matrix !
            gradient += 2 * (output(j) - target(j)) * weights(j, i) * indicatrice(output(j));
        }
        // layerGradient.push_back(gradient);
    }

    // Calcul du gradient de l'erreur selon le poids pour la couche l
    xt::xarray<float> weightsGradient = xt::empty<float>({weights.shape()[0], weights.shape()[1]});
    for (int i = 0; i < weights.shape()[0]; ++i)
    {
        float gradient = 0.0;
        for (int j = 0; j < weights.shape()[1]; ++j)
        {
            weightsGradient(i, j) += layerGradient(j) * indicatrice(output(j));
        }
    }

    // Mise a jour des poids lie a la couche l
    for (int i = 0; i < weights.shape()[0]; ++i)
    {
        for (int j = 0; j < weights.shape()[1]; ++j)
        {
            weights(i, j) -= tauxApprentissage * weightsGradient(i, j) * output(j);
        }
    }
}
void Dense::dropout(u_int8_t dropRate)
{
    std::random_device rd;
    std::mt19937 gen(rd());

    for (int i = 0; i < this->weights.shape()[0]; ++i)
    {
        for (int j = 0; j < this->weights.shape()[1]; ++j)
        {
            if (dropRate <= std::uniform_int_distribution<>(1, 100)(gen))
            {
                this->weights(i, j) = 0.0;
            }
        }
    }

    std::cout << "          | dropout p=" << dropRate
            << "\n          v" << std::endl;
}