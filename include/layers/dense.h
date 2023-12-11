#ifndef DENSE_H
#define DENSE_H

#include "layer.h"

// Dense(int depth, std::tuple<int, int, int> inputShape, std::tuple<int, int, int, int, int> weightsShape)
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

        float beta = 0.0;
        float gamma = 1.0;

        Dense(int inputShape, int outputShape)
        {

            this->inputShape = inputShape;
            this->outputShape = outputShape;
            this->weightsShape = std::tuple<int, int>{inputShape, outputShape};

            this->output = xt::empty<float>({outputShape});
            this->input = xt::empty<float>({inputShape});

            // weights = kernelsGaussianDistro(1, 1, inputShape, outputShape);

            weights = xt::random::randn<float>({inputShape, outputShape});
            drop = xt::empty<bool>({inputShape, outputShape});
        }

        ~Dense() = default;

        void forward(xt::xarray<float> input) override;

        void backward(
            xt::xarray<float> target,
            float learningRate);

        void dropout(uint16_t dropRate);
};

#endif