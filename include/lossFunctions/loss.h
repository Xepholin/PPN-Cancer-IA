#ifndef LOSS_H
#define LOSS_H

#include <iostream>
#include <xtensor/xarray.hpp>

enum LossType {
	LOSS_MSE,
	LOSS_CROSS_ENTROPY
};

std::ostream& operator<<(std::ostream& out, const LossType value);

#define mse LossType::LOSS_MSE
#define cross_entropy LossType::LOSS_CROSS_ENTROPY

class Loss {
   public:
	std::string name = "LossFunction";

	virtual float compute(xt::xarray<float> output, xt::xarray<int> label);
	virtual xt::xarray<float> prime(xt::xarray<float> output, xt::xarray<int> label);
};

#endif