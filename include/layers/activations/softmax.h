#ifndef SOFTMAX_H
#define SOFTMAX_H

#include "activation.h"

class Softmax : public Activation
{
public:
    int depth = 0;

    // 1 x Longueur
    int inputShape = 0;
    int outputShape = 0;

    // Depth - Height
    std::tuple<int, int> weightsShape{0, 0};

    // Depth - Height
    xt::xarray<float> weights;

    float beta = 0;
    float gamma = 1;

    Softmax(int inputShape, int outputShape)
    {

        this->inputShape = inputShape;
        this->outputShape = outputShape;
        this->weightsShape = std::tuple<int, int>{inputShape, outputShape};

        this->output = xt::empty<float>({outputShape});
        this->input = xt::empty<float>({inputShape});

        weights = xt::random::rand<float>({inputShape, outputShape}, 0, 1);
    }

    ~Softmax() = default;

    void forward(xt::xarray<float> input) override;

    void backward(xt::xarray<float> gradient) override;
};

#endif