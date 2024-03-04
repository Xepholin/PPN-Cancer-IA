#include "loss.h"

float Loss::compute(xt::xarray<float> output, xt::xarray<int> label)	{
	std::cout << "compute loss" << std::endl;
}

xt::xarray<float> Loss::prime(xt::xarray<float> output, xt::xarray<int> label)	{
	std::cout << "prime loss" << std::endl;
}