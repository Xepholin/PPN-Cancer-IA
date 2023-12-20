#include <iostream>
#include <random>

#include "dense.h"
#include "tools.h"

void Dense::forward(xt::xarray<float> input)
{   
    if (this->flatten)  {
        this->input = xt::flatten(input);
    }
    else    {
        this->input = input;
    }

    std::cout << "input\n" << this->input << '\n' << std::endl;
    std::cout << "weights\n" << this->weights << '\n' << std::endl;
    
    for (int j = 0; j < this->weights.shape()[1]; ++j)
    {
        float dotResult = 0;
        for (int i = 0; i < this->weights.shape()[0]; ++i)
        {
            if (drop(i) == true)
            {
                continue;
            }

            dotResult += weights(i, j) * this->input(i);
        }
        output(j) = dotResult;
    }

    // std::cout << "before batchnorm output\n" << this->output << '\n' << std::endl;

    if (this->activationType != softmax)    {
        this->output = batchNorm(this->output, this->beta, this->gamma);
    }

    // std::cout << "before relu output\n" << this->output << '\n' << std::endl;

    if (this->activationType != ActivationType::ACTIVATION_NO_TYPE) {
        this->activation->forward(this->output);
        this->output = this->activation->output;   
    }

    std::cout << "output\n" << this->output << '\n' << std::endl;
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
            gradient +=  2.0*(cost - output(j))* this->weights(i, j) * this->activation->prime(output(j));
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

            this->weights(i, j) -= learningRate * weightsGradient(i, j) * this->input(i);
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

void Dense::heWeightsInit()    {
    float std = sqrt(2.0 / (static_cast<float>(this->inputShape)));

    this->weights = xt::random::randn<float>({this->inputShape, this->outputShape}, 0, std/1000.0);
}

void Dense::XGWeightsInit() {
    float std = sqrt(2.0 / (static_cast<float>(this->inputShape) + this->outputShape));

    this->weights = xt::random::randn<float>({this->inputShape, this->outputShape}, 0, std/1000.0);
}