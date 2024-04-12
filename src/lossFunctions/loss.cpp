#include "loss.h"

float Loss::compute(xt::xarray<float> output __attribute__((unused)), xt::xarray<int> label __attribute__((unused)))	{
	std::cout << "compute loss" << std::endl;
	return 0.0;
}

xt::xarray<float> Loss::prime(xt::xarray<float> output __attribute__((unused)), xt::xarray<int> label __attribute__((unused)))	{
	std::cout << "prime loss" << std::endl;
	return 0;
}