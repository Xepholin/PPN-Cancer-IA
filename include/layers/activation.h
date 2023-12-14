#ifndef ACTIVATION_H
#define ACTIVATION_H

#include "layer.h"

enum ActivationType {
    ACTIVATION_NO_TYPE,
    ACTIVATION_RELU,
    ACTIVATION_SOFTMAX
};

std::ostream& operator<<(std::ostream& out, const ActivationType value);

#define relu ActivationType::ACTIVATION_RELU
#define softmax ActivationType::ACTIVATION_SOFTMAX

// Activation()
class Activation : public ILayer
{

    public:

        Activation()    {
            this->name = "Activation";
        }

        ~Activation() = default;
    
        virtual void forward(xt::xarray<float> input) override;

        virtual void backward(xt::xarray<float> cost, float learningRate) override;

        virtual float prime(float x);

        virtual void print() const override;
};

#endif