#ifndef CROSS_ENTROPY_H
#define CROSS_ENTROPY_H

#include "loss.h"

class CrossEntropy : public Loss {
   public:
	int labelSize = 0;

	CrossEntropy() {
		name = "Cross Entropy";
	}

	virtual float compute(xt::xarray<float> output, xt::xarray<int> label) override;
	virtual float prime(float output, int label) override;
};

#endif