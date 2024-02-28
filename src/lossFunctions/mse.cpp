#include "mse.h"

float MSE::compute(xt::xarray<float> output, xt::xarray<int> label) {
	float err = 0.0;
	for (int i = 0; i < output.size(); ++i) {
		err += ((output(i) - label(i)) * (output(i) - label(i)));
	}

	err *= 1.0 / label.size();

	return err;
}

float MSE::prime(float output, int label) {
	return (2.0 * (output - label));
}