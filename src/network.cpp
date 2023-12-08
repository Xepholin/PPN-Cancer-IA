#include <iostream>

#include "network.h"
#include <random>

void ILayer::forward(xt::xarray<float> input)
{
    std::cout << "ILayer forward" << std::endl;
}

void ILayer::backward(xt::xarray<float> gradient)
{
    std::cout << "ILayer backward" << std::endl;
}

//

void ConvolutionLayer::forward(xt::xarray<float> input)
{
    this->input = input;

    std::cout << "Convolution forward" << std::endl;
    for (int i = 0; i < this->filters.shape()[0]; ++i)
    {

        for (int j = 0; j < this->filters.shape()[1]; ++j)
        {
            auto tmpMat = xt::view(input, j);
            auto tmpFilter = xt::view(this->filters, i, j);
            auto convolution_result = matrixConvolution(tmpMat, tmpFilter, std::get<3>(this->filtersShape), std::get<4>(this->filtersShape));

            xt::view(output, i) = convolution_result;
        }
    }
}

void ConvolutionLayer::backward(xt::xarray<float> gradient)
{
    std::cout << "Convolution backward" << std::endl;
}

//

void PoolingLayer::forward(xt::xarray<float> input)
{
    std::cout << "Pooling forward" << std::endl;

    this->input = input;

    for (int i = 0; i < this->output.shape()[0]; ++i)
    {

        auto tmpMat = xt::view(input, i);
        auto convolution_result = poolingMatrice(tmpMat);
        xt::view(output, i) = convolution_result;
    }
}

void PoolingLayer::backward(xt::xarray<float> gradient)
{
    std::cout << "Pooling backward" << std::endl;
}

xt::xarray<float> PoolingLayer::poolingMatrice(xt::xarray<float> matrix)
{

    int padding = this->padding;
    int stride = this->stride;
    int sizePooling = this->size;

    int sizeNewMatriceX = std::get<1>(this->outputShape);
    int sizeNewMatriceY = std::get<2>(this->outputShape);

    if (this->padding > 0)
    {
        matrix = padMatrice(matrix, padding);
    }

    xt::xarray<float> pooledMatrix{xt::empty<uint8_t>({sizeNewMatriceX, sizeNewMatriceY})};

    int incr = stride - 1;

    for (int i = 0; i < sizeNewMatriceX; ++i)
    {

        for (int j = 0; j < sizeNewMatriceY; ++j)
        {
            xt::xrange<int> rows(i + i * incr, i + i * incr + sizePooling);
            xt::xrange<int> cols(j + j * incr, j + j * incr + sizePooling);

            auto a = xt::view(matrix, rows, cols);

            pooledMatrix(i, j) = this->pooling(a);
        }
    }

    return pooledMatrix;
}

float PoolingLayer::pooling(xt::xarray<float> matrix)
{

    float result = -1.0; // Default value if an invalid poolingType is provided

    switch (this->type)
    {
    case PoolingLayer::PoolingType::NO_TYPE:
    {
        result = 0.0;
        break;
    }
    case PoolingLayer::PoolingType::MIN:
    {
        result = *std::min_element(matrix.begin(), matrix.end());
        break;
    }
    case PoolingLayer::PoolingType::MAX:
    {
        result = *std::max_element(matrix.begin(), matrix.end());
        break;
    }
    case PoolingLayer::PoolingType::AVG:
    {
        result = xt::sum(matrix)() / static_cast<float>(matrix.size());
        break;
    }

    default:
    {
        break;
    }
    }

    return result;
}

//

void ActivationLayer::forward(xt::xarray<float> input)
{
    std::cout << "Activation forward" << std::endl;
}

void ActivationLayer::backward(xt::xarray<float> gradient)
{
    std::cout << "Activation backward" << std::endl;
}

xt::xarray<float> ActivationLayer::activation(xt::xarray<float> matrix)
{
    std::cout << "Ici ca active ConvolutionLayer" << std::endl;
    return 0.0;
}

//

void DenseLayer::forward(xt::xarray<float> input)
{
    std::cout << "forward Dense" << std::endl;

    this->input = input;

    for (int i = 0; i < this->weights.shape()[0]; ++i)
    {
        for (int j = 0; j < this->weights.shape()[1]; ++j)
        {
            output(j) += this->weights(i, j) * this->input(i);
        }
    }
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

void DenseLayer::backward(
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

void DenseLayer::backwardHiddenLayer(
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
void DenseLayer::dropout(u_int8_t dropRate)
{
    std::cout << "dropout Dense" << std::endl;

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
}

xt::xarray<float> flatten(xt::xarray<float> input)
{
    return xt::flatten(input);
}