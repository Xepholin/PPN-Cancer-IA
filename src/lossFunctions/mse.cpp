#include "mse.h"

#include <xtensor/xrandom.hpp>
#include <xtensor/xio.hpp>

float MSE::compute(xt::xarray<float> output, xt::xarray<int> label) {
	float err = 0.0;
	for (int i = 0; i < output.size(); ++i) {
		err += ((output(i) - label(i)) * (output(i) - label(i)));
	}

	err *= 1.0 / label.size();

	return err;
}

xt::xarray<float> MSE::prime(xt::xarray<float> output, xt::xarray<int> label) {
	int outputSize = output.size();
	xt::xarray<float> prime = xt::empty<float>({outputSize});

	for (int i = 0; i < outputSize; ++i) {
		prime(i) = (2.0 * (output(i) - label(i)));
	}

	return prime;
}