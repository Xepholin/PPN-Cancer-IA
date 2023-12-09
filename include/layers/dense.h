#ifndef DENSE_H
#define DENSE_H

#include "layer.h"

// Dense(int depth, std::tuple<int, int, int> inputShape, std::tuple<int, int, int, int, int> weightsShape)
class Dense : public ILayer
{
    public:
        int depth = 0;

        // 1 x Longueur
        int inputShape = 0;
        int outputShape = 0;

        // Height -Width
        std::tuple<int, int> weightsShape{0, 0};

        // Height -Width
        xt::xarray<float> weights;
        xt::xarray<float> tmpWeights;

        // 1D
        xt::xarray<float> input;
        xt::xarray<float> output;

        int bias = 1;

        Dense(int inputShape, int outputShape)
        {

            this->inputShape = inputShape;
            this->outputShape = outputShape;
            this->weightsShape = std::tuple<int, int>{inputShape, outputShape};

            this->output = xt::empty<float>({outputShape});
            this->input = xt::empty<float>({inputShape});

            weights = xt::random::rand<float>({inputShape, outputShape}, 0, 1);
        }

        ~Dense() = default;

        void forward(xt::xarray<float> input) override;

        void backward(/*xt::xarray<float> gradient, */
                    xt::xarray<float> target,
                    xt::xarray<float> outputL,
                    xt::xarray<float> weights2,
                    xt::xarray<float> weights1,
                    xt::xarray<float> layerBefore,
                    float tauxApprentissage);

        void backward(
            xt::xarray<float> target,
            xt::xarray<float> outputL,
            xt::xarray<float> weightsL,
            float tauxApprentissage);

        void backwardHiddenLayer(
            xt::xarray<float> target,
            xt::xarray<float> output,
            xt::xarray<float> weights,
            float tauxApprentissage);

        void dropout(u_int8_t dropRate);
};

#endif