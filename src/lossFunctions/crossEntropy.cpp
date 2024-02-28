#include "crossEntropy.h"

float CrossEntropy::compute(xt::xarray<float> output, xt::xarray<int> label) {
	labelSize = label.size();
	
	float err = 0.0;

	for (int i = 0; i < output.size(); ++i) {
		// on suppose ici que label est encodé en one-hot, où seulement un élément est 1, les autres sont 0
		if (label(i) == 1) {
			err -= std::log(output(i) + 1e-9);
			// Ajoute un petit nombre 1e-9 pour éviter un log négatif infini
		}
	}
	return err;
}

float CrossEntropy::prime(float output, int label) {
	return (2.0 * (output - label));
}