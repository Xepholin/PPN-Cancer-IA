#include <iostream>

#include <xtensor/xview.hpp>

#include "convolution.h"
#include "conv_op.h"
#include "tools.h"

void Convolution::forward(xt::xarray<float> input)
{
    this->input = input;

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

    this->output = batchNorm(this->output, this->beta, this->gamma);

    // std::cout << "Convolution with " <<
    // this->filters.shape()[2] << "x" << this->filters.shape()[3] << " kernel" <<
    // " + " << std::get<3>(this->filtersShape) << " stride" <<
    // " + " << std::get<4>(this->filtersShape) << " pad : " <<
    // this->output.shape()[1] << "x" << this->output.shape()[2] << "x" << this->output.shape()[0] <<
    // "\n          |\n          v" << std::endl;
}

void Convolution::backward(xt::xarray<float> gradient, float learningRate)
{
    std::cout << "Convolution backward" << std::endl;
}