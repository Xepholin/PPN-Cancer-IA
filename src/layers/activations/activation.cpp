#include <iostream>

#include "activation.h"

void Activation::forward(xt::xarray<float> input)
{
    std::cout << "Activation forward" << std::endl;
}

xt::xarray<float> Activation::backward(xt::xarray<float> cost, float learningRate)
{
    std::cout << "Activation backward" << std::endl;
	return 0;
}

xt::xarray<float> Activation::prime(xt::xarray<float> input)    {
    std::cout << "Activation prime" << std::endl;
    return input;
}

void Activation::print() const
{
    std::cout << "print Activation" << std::endl;
}

std::ostream& operator<<(std::ostream& out, const ActivationType value)
{
    switch (value)
    {
        case ActivationType::ACTIVATION_NO_TYPE:
            return out << "no_type.";
        case ActivationType::ACTIVATION_RELU:
            return out << "relu.";
        case ActivationType::ACTIVATION_SOFTMAX:
            return out << "softmax.";
        case ActivationType::ACTIVATION_SIGMOID:
            return out << "sigmoid.";
        default:
            return out << "unknown type.";
    }
}