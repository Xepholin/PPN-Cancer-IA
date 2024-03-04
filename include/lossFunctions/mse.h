#ifndef MSE_H
#define MSE_H

#include "loss.h"

class MSE : public Loss {
   public:
	MSE() {
		name = "MSE";
	}

	~MSE() = default;

	virtual float compute(xt::xarray<float> output, xt::xarray<int> label) override;
	virtual xt::xarray<float> prime(xt::xarray<float> output, xt::xarray<int> label) override;
};

#endif