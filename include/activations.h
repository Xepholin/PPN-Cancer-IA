#ifndef ACTIVATIONS_H
#define ACTIVATIONS_H

#include "network.h"

class ReLu : public ActivationLayer {
    public:

        ReLu()  {
            
        }

        void forward(xt::xarray<float> input) override;
        
        void backward(xt::xarray<float> gradient) override;

        xt::xarray<float> activation(xt::xarray<float> matrix) override;
};

#endif