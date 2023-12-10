#ifndef DENSE_H
#define DENSE_H

#include <xtensor/xarray.hpp>
#include <xtensor/xrandom.hpp>

// Dense(int depth, std::tuple<int, int, int> inputShape, std::tuple<int, int, int, int, int> weightsShape)
class Dense
{
    public:

        xt::xarray<float> input;
        xt::xarray<float> output;

        // 1 x Longueur
        int inputShape = 0;
        int outputShape = 0;

        // Height -Width
        std::tuple<int, int> weightsShape{0, 0};

        // Height -Width
        xt::xarray<float> weights;

        xt::xarray<bool> drop;

        int bias = 1;

        Dense(int inputShape, int outputShape)
        {

            this->inputShape = inputShape;
            this->outputShape = outputShape;
            this->weightsShape = std::tuple<int, int>{inputShape, outputShape};

            this->output = xt::empty<float>({outputShape});
            this->input = xt::empty<float>({inputShape});

            weights = xt::random::rand<float>({inputShape, outputShape}, 0, 1);
            drop = xt::empty<bool>({inputShape, outputShape});
        }

        ~Dense() = default;

        void forward(xt::xarray<float> input);

        void backward(
            xt::xarray<float> target,
            float tauxApprentissage);

        void dropout(uint8_t dropRate);
};

#endif