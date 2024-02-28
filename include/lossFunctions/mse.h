#ifndef MSE_H
#define MSE_H

#include "loss.h"

class MSE : public Loss {
   public:
	MSE() {
		name = "MSE";
	}

	virtual float compute(xt::xarray<float> output, xt::xarray<int> label) override;
	virtual float prime(float output, int label) override;
};

#endif