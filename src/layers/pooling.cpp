#include <iostream>

#include "pooling.h"
#include "conv_op.h"

void Pooling::forward(xt::xarray<float> input)
{
    this->input = input;

    for (int i = 0; i < this->output.shape()[0]; ++i)
    {
        xt::xarray<float> tmpMat = xt::view(input, i);
        xt::xarray<float> convolution_result = poolingMatrice(tmpMat);
        xt::view(output, i) = convolution_result;
    }
}

xt::xarray<float> Pooling::backward(xt::xarray<float> cost, float learningRate)
{
    std::cout << "Pooling backward" << std::endl;
}

void Pooling::print() const
{
    std::cout << "Pool with " <<
    this->size << "x" << this->size << ' ' << this->type << " kernel" << 
    " + " << this->stride << " stride" <<
    " + " << this->padding << " pad : " <<
    this->output.shape()[1] << "x" << this->output.shape()[2] << "x" << this->output.shape()[0] <<
    "\n          |\n          v" << std::endl;
}

xt::xarray<float> Pooling::poolingMatrice(xt::xarray<float> matrix)
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

            xt::xarray<float> a = xt::view(matrix, rows, cols);

            pooledMatrix(i, j) = this->pooling(a);
        }
    }

    return pooledMatrix;
}

float Pooling::pooling(xt::xarray<float> matrix)
{

    float result = -1.0; // Default value if an invalid poolingType is provided

    switch (this->type)
    {
        case PoolingType::POOLING_NO_TYPE:
        {
            result = 0.0;
            break;
        }
        case PoolingType::POOLING_MIN:
        {
            result = *std::min_element(matrix.begin(), matrix.end());
            break;
        }
        case PoolingType::POOLING_MAX:
        {
            result = *std::max_element(matrix.begin(), matrix.end());
            break;
        }
        case PoolingType::POOLING_AVG:
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

std::ostream& operator<<(std::ostream& out, const PoolingType value)
{
    switch (value)
    {
        case PoolingType::POOLING_NO_TYPE:
            return out << "no_type.";
        case PoolingType::POOLING_MAX:
            return out << "max.";
        case PoolingType::POOLING_MIN:
            return out << "min.";
        case PoolingType::POOLING_AVG:
            return out << "avg.";
        default:
            return out << "unknown type.";
    }
}