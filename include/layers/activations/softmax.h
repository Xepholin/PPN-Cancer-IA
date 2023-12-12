#ifndef SOFTMAX_H
#define SOFTMAX_H

#include <tuple>

#include "activation.h"

class Softmax : public Activation
{
public:

    
    std::tuple<int, int, int> inputShape{0, 0, 0};
    std::tuple<int, int, int> outputShape{0, 0, 0};
    
    float beta = 0;
    float gamma = 1;

    Softmax(std::tuple<int, int, int> inputShape)
    {
        this->inputShape = inputShape;
        this->outputShape = inputShape;

        int depth = std::get<0>(inputShape);
        int height = std::get<1>(inputShape);
        int width = std::get<2>(inputShape);

        this->input = xt::empty<float>({depth, height, width});
        this->output = xt::empty<float>({depth, height, width});
    }

    ~Softmax() = default;

    virtual void forward(xt::xarray<float> input) override;

    virtual void backward(xt::xarray<float> cost, float learningRate) override;

    virtual float prime(float x) override;

    void print() const override;

    xt::xarray<float> softmaxGradient();
};

#endif