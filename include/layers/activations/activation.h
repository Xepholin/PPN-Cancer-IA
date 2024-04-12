#ifndef ACTIVATION_H
#define ACTIVATION_H

#include "layer.h"

enum ActivationType {
    ACTIVATION_NO_TYPE,
    ACTIVATION_RELU,
    ACTIVATION_SOFTMAX,
    ACTIVATION_SIGMOID
};

std::ostream& operator<<(std::ostream& out, const ActivationType value);

#define relu ActivationType::ACTIVATION_RELU
#define softmax ActivationType::ACTIVATION_SOFTMAX
#define sigmoid ActivationType::ACTIVATION_SIGMOID

// Activation()
class Activation : public ILayer
{

    public:

        Activation()	{
			name = "Activation";
		}

        ~Activation() = default;
    
        virtual void forward(xt::xarray<float> input, bool training = true) override;

        virtual xt::xarray<float> backward(xt::xarray<float> gradient, float learningRate) override;

        virtual xt::xarray<float> prime(xt::xarray<float> input);

        virtual void print() const override;
};

#endif