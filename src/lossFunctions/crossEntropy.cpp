#include "crossEntropy.h"

#include <xtensor/xrandom.hpp>

float CrossEntropy::compute(xt::xarray<float> output, xt::xarray<int> label) {
	float err = 0.0;
	int labelSize = label.size();

	if (labelSize == 2) {
		for (int i = 0; i < output.size(); ++i) {
			err += -(label(i) * std::log(output(i)) + (1 - label(i)) * std::log(1 - output(i)));
		}

		err *= 1.0 / label.size();
	} else if (labelSize > 2) {
		for (int i = 0; i < output.size(); ++i) {
			err += (label(i) * std::log(output(i)));
		}

		err = -err;
	} else {
		perror("Wrong label size Compute");
	}

	return err;
}

xt::xarray<float> CrossEntropy::prime(xt::xarray<float> output, xt::xarray<int> label) {
	int outputSize = output.size();
	int labelSize = label.size();
	xt::xarray<float> prime = xt::empty<float>({outputSize});

	if (labelSize == 2) {
		for (int i = 0; i < output.size(); ++i) {
			prime(i) = -(label(i) / output(i)) + ((1 - label(i)) / (1 - output(i)));
		}
	} else if (labelSize > 2) {
		for (int i = 0; i < output.size(); ++i) {
			prime(i) = -(label(i) / output(i));
		}
	} else {
		perror("Wrong label size Prime");
	}

	return prime;
}