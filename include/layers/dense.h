#ifndef DENSE_H
#define DENSE_H

#include "layer.h"
#include "activation.h"
#include "relu.h"
#include "softmax.h"

// Dense(int inputShape, int outputShape)
class Dense : public ILayer
{
    public:
    
        // 1 x Longueur
        int inputShape = 0;
        int outputShape = 0;

        // Height -Width
        std::tuple<int, int> weightsShape{0, 0};

        // Height -Width
        xt::xarray<float> weights;

        xt::xarray<bool> drop;

        int bias = 1;

        ActivationType activationType = ActivationType::ACTIVATION_NO_TYPE;
        Activation *activation;

        float beta = 0.0;
        float gamma = 1.0;

        Dense(int inputShape, int outputShape, ActivationType activationType = ActivationType::ACTIVATION_NO_TYPE)
        {
            this->name = "Dense";

            this->inputShape = inputShape;
            this->outputShape = outputShape;
            this->weightsShape = std::tuple<int, int>{inputShape, outputShape};

            this->output = xt::empty<float>({outputShape});
            this->input = xt::empty<float>({inputShape});


            this->weights = xt::random::randn<float>({inputShape, outputShape} );
            drop = xt::empty<bool>({inputShape});

            this->activationType = activationType;

            switch (this->activationType)
            {
                case ActivationType::ACTIVATION_NO_TYPE:
                    this->activation = new Activation;
                    break;
                case ActivationType::ACTIVATION_RELU:
                    this->activation = new ReLu(std::tuple<int, int ,int>{1, 1, outputShape});
                    break;
                case ActivationType::ACTIVATION_SOFTMAX:
                    this->activation = new Softmax(outputShape);
                    break;
                default:
                    perror("Dense Activation Type Error");
            }
        }

        ~Dense()    {
            delete this->activation;
        }

        virtual void forward(xt::xarray<float> input) override;

        virtual void backward(
            float cost,
            float learningRate);

        void print() const override;

        void dropout(uint16_t dropRate);

        void printDropout(uint16_t dropRate) const;
};

#endif