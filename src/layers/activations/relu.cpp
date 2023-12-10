#include <iostream>

#include "relu.h"

void ReLu3D::forward(xt::xarray<float> input)
{
    this->output = xt::where(input <= 0, 0.0, input);

    std::cout << "          | ReLu\n"
              << "          v" << std::endl;
}

void ReLu3D::backward()
{
    std::cout << "ReLu backward" << std::endl;
}

void ReLu3D::batchNorm()
{
    int total = xt::sum(this->output)();
    int size = this->output.size();
    float mean = total / size;
    float stddev = xt::stddev(this->output)();

    this->output = (this->output - mean) / stddev;

    this->output *= this->gamma;
    this->output += this->beta;
}

void ReLu1D::forward(xt::xarray<float> input)
{
    this->output = xt::where(input <= 0, 0.0, input);

    std::cout << "          | ReLu\n"
              << "          v" << std::endl;
}

void ReLu1D::backward()
{
    std::cout << "ReLu1D backward" << std::endl;
}