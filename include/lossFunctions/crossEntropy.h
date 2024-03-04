#ifndef CROSS_ENTROPY_H
#define CROSS_ENTROPY_H

#include "loss.h"

class CrossEntropy : public Loss {
   public:
	int labelSize = 0;

	CrossEntropy() {
		name = "Cross Entropy";
	}

	~CrossEntropy() = default;

	virtual float compute(xt::xarray<float> output, xt::xarray<int> label) override;
	virtual xt::xarray<float> prime(xt::xarray<float> output, xt::xarray<int> label) override;
};

#endif