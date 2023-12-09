#ifndef SOFTMAX_H
#define SOFTMAX_H

#include "activation.h"

class Softmax1D : public Activation
{
public:
    int depth = 0;

    // 1 x Longueur
    int inputShape = 0;
    int outputShape = 0;

    float beta = 0;
    float gamma = 1;

    Softmax1D(int inputShape)
    {

        this->inputShape = inputShape;
        this->outputShape = inputShape;

        this->output = xt::empty<float>({outputShape});
        this->input = xt::empty<float>({inputShape});
    }

    ~Softmax1D() = default;

    void forward(xt::xarray<float> input) override;

    void backward(xt::xarray<float> gradient) override;
};

#endif