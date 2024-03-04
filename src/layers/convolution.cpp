#include <iostream>

#include "convolution.h"
#include "conv_op.h"
#include "tools.h"

void Convolution::forward(xt::xarray<float> input)
{
    this->input = input;
    // std::cout << "input\n" << this->input << '\n' << std::endl;
    // std::cout << "filters\n" << this->filters << '\n' << std::endl;

    for (int i = 0; i < this->filters.shape()[0]; ++i)
    {

        for (int j = 0; j < this->filters.shape()[1]; ++j)
        {
            xt::xarray<float> tmpMat = xt::view(input, j);
            xt::xarray<float> tmpFilter = xt::view(this->filters, i, j);
            xt::xarray<float> convolution_result = matrixConvolution(tmpMat, tmpFilter, std::get<3>(this->filtersShape), std::get<4>(this->filtersShape));

            xt::view(output, i) = convolution_result;
        }
    }

	if (this->normalize)
	{
		this->output = normalized(this->output);
	}

    if (this->activation->name != "Activation") {
        this->activation->forward(this->output);
        this->output = this->activation->output;
    }

    // std::cout << "output\n" << this->output << '\n' << std::endl;
}

xt::xarray<float> Convolution::backward(xt::xarray<float> cost, float learningRate)
{
    std::cout << "Convolution backward" << std::endl;
}

void Convolution::print() const
{
    std::cout << "Convolution with " <<
    this->filters.shape()[2] << "x" << this->filters.shape()[3] << " kernel" <<
    " + " << std::get<3>(this->filtersShape) << " stride" <<
    " + " << std::get<4>(this->filtersShape) << " pad : " <<
    this->output.shape()[1] << "x" << this->output.shape()[2] << "x" << this->output.shape()[0] <<
    "\n          |\n          v" << std::endl;
}

void Convolution::heWeightsInit()   {
    int inputHeight = std::get<1>(this->inputShape);
    int inputWidth = std::get<2>(this->inputShape);

    int filtersDepth = std::get<0>(this->filtersShape);
    int filtersHeight = std::get<1>(this->filtersShape);
    int filtersWidth = std::get<2>(this->filtersShape);

    float std = sqrt(2.0 / (static_cast<float>(filtersDepth) * this->depth * inputHeight * inputWidth));
    
    this->filters = xt::random::randn<float>({filtersDepth, depth, 
                                              filtersHeight, filtersWidth}, 0, std);
}

void Convolution::XGWeightsInit()   {
    int inputDepth = std::get<0>(this->inputShape);
    int inputHeight = std::get<1>(this->inputShape);
    int inputWidth = std::get<2>(this->inputShape);

    int filtersDepth = std::get<0>(this->filtersShape);
    int filtersHeight = std::get<1>(this->filtersShape);
    int filtersWidth = std::get<2>(this->filtersShape);

    float std = sqrt(2.0 / (static_cast<float>(this->input.size()) + this->output.size()));
    this->filters = xt::random::randn<float>({filtersDepth, depth, 
                                              filtersHeight, filtersWidth}, 0, std);
}