#ifndef SOFTMAX_H
#define SOFTMAX_H

#include <tuple>

#include "activation.h"

class Softmax : public Activation
{
public:

    
    int inputShape = 0;
    int outputShape = 0;
    
    float beta = 0;
    float gamma = 1;

    Softmax(int inputShape)
    {
        this->name = "Softmax";
        this->inputShape = inputShape;
        this->outputShape = inputShape;

        this->input = xt::empty<float>({inputShape});
        this->output = xt::empty<float>({inputShape});
    }

    ~Softmax() = default;

    virtual void forward(xt::xarray<float> input) override;

    virtual void backward(xt::xarray<float> cost, float learningRate) override;

    virtual float prime(float x) override;

    void print() const override;

    xt::xarray<float> softmaxGradient();
};

#endif