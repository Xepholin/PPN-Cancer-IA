#include "crossEntropy.h"

#include <math.h>

#include <xtensor/xrandom.hpp>

float CrossEntropy::compute(xt::xarray<float> output, xt::xarray<int> label) {
	float err = 0.0;
	int labelSize = label.size();

	if (labelSize == 2) {
		for (int i = 0; i < labelSize; ++i) {
			err += -(label(i) * logf(output(i)) + (1.0 - label(i)) * logf(1.0 - output(i)));
		}

		err *= 1.0 / labelSize;
	} else if (labelSize > 2) {
		for (int i = 0; i < labelSize; ++i) {
			err += (label(i) * logf(output(i)));
		}

		err = -err;
	} else {
		perror("Wrong label size Compute");
		exit(0);
	}

	return err;
}

xt::xarray<float> CrossEntropy::prime(xt::xarray<float> output, xt::xarray<int> label) {
	int outputSize = output.size();
	int labelSize = label.size();
	xt::xarray<float> prime = xt::empty<float>({outputSize});

	if (labelSize == 2) {
		for (int i = 0; i < outputSize; ++i) {
			prime(i) = -(label(i) / output(i)) + ((1.0 - label(i)) / (1.0 - output(i)));
		}
	} else if (labelSize > 2) {
		for (int i = 0; i < outputSize; ++i) {
			prime(i) = -(label(i) / output(i));
		}
	} else {
		perror("Wrong label size Prime");
		exit(0);
	}

	return prime;
}