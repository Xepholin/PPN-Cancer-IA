#ifndef SOFTMAX_H
#define SOFTMAX_H

#include "activation.h"

class Softmax2D : public Activation
{
public:

    int depth = 0;

    // 1 x Longueur
    int inputShape = 0;
    int outputShape = 0;

    float beta = 0;
    float gamma = 1;

    Softmax2D(int inputShape)
    {

        this->inputShape = inputShape;
        this->outputShape = inputShape;

        this->output = xt::empty<float>({outputShape});
        this->input = xt::empty<float>({inputShape});
    }

    ~Softmax2D() = default;

    void forward(xt::xarray<float> input) override;

    void backward() override;

    xt::xarray<float> softmaxGradient();
};

#endif