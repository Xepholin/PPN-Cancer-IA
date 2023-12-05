#include <iostream>

#include "network.h"

void ILayer::forward(xt::xarray<float> input)
{
    std::cout << "ILayer forward" << std::endl;
}

void ILayer::backward(xt::xarray<float> gradient)
{
    std::cout << "ILayer backward" << std::endl;
}

float ILayer::pooling(xt::xarray<float> matrix)
{
    std::cout << "Ici ca pool ILayer" << std::endl;
    return 0.0;
}

xt::xarray<float> ILayer::poolingMatrice(xt::xarray<float> matrix)
{
    std::cout << "Ici ca poolMatrice ILayer" << std::endl;
    return matrix;
}

float ILayer::activation(xt::xarray<float> matrix)
{
    std::cout << "Ici ca active ILayer" << std::endl;
    return 0.0;
}

//

void ConvolutionLayer::forward(xt::xarray<float> input)
{
    std::cout << "Convolution forward" << std::endl;
}

void ConvolutionLayer::backward(xt::xarray<float> gradient)
{
    std::cout << "Convolution backward" << std::endl;
}

xt::xarray<float> ConvolutionLayer::poolingMatrice(xt::xarray<float> matrix)
{

    int padding = this->pool.padding;
    int stride = this->pool.stride;
    int sizePooling = this->pool.size;

    int sizeNewMatriceX = (matrix.shape()[0] - sizePooling + 2 * padding) / stride + 1;
    int sizeNewMatriceY = (matrix.shape()[1] - sizePooling + 2 * padding) / stride + 1;

    if (this->pool.padding > 0)
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

            // std::cout << "Mat travail" << std::endl;
            // std::cout << a << std::endl;

            pooledMatrix(i, j) = this->pooling(a);
        }
    }

    return pooledMatrix;
}

float ConvolutionLayer::pooling(xt::xarray<float> matrix)
{

    float result = -1.0; // Default value if an invalid poolingType is provided

    switch (this->pool.type)
    {
    case Pooling::PoolingType::NO_TYPE:
    {
        result = 0.0;
        break;
    }
    case Pooling::PoolingType::MIN:
    {
        result = *std::min_element(matrix.begin(), matrix.end());
        break;
    }
    case Pooling::PoolingType::MAX:
    {
        result = *std::max_element(matrix.begin(), matrix.end());
        break;
    }
    case Pooling::PoolingType::AVG:
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

float ConvolutionLayer::activation(xt::xarray<float> matrix)
{
    std::cout << "Ici ca active ConvolutionLayer" << std::endl;
    return 0.0;
}