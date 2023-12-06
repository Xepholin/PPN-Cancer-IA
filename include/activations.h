#ifndef ACTIVATIONS_H
#define ACTIVATIONS_H

#include "network.h"

class ReLu : public ActivationLayer {
    public:
        
        void forward(xt::xarray<float> input) override;

        void backward(xt::xarray<float> gradient) override;

        float activation(xt::xarray<float> matrix) override;
};

#endif