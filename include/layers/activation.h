#ifndef ACTIVATION_H
#define ACTIVATION_H

#include "layer.h"

// Activation()
class Activation : public ILayer
{

    public:
    
        void forward(xt::xarray<float> input) override;

        void backward(xt::xarray<float> gradient, float learningRate) override;
};

#endif