#ifndef OUTPUTLAYER_H
#define OUTPUTLAYER_H

#include <xtensor/xarray.hpp>
#include <xtensor/xrandom.hpp>

// outputLayer(int depth, std::tuple<int, int, int> inputShape, std::tuple<int, int, int, int, int> weightsShape)
class outputLayer
{
    public:

        xt::xarray<float> input;
        xt::xarray<float> output;

        int depth = 0;

        // 1 x Longueur
        int inputShape = 0;
        int outputShape = 0;

        // Height -Width
        std::tuple<int, int> weightsShape{0, 0};

        // Height -Width
        xt::xarray<float> weights;

        xt::xarray<bool> drop;

        int bias = 1;

        outputLayer(int inputShape, int outputShape)
        {

            this->inputShape = inputShape;
            this->outputShape = outputShape;
            this->weightsShape = std::tuple<int, int>{inputShape, outputShape};

            this->output = xt::empty<float>({outputShape});
            this->input = xt::empty<float>({inputShape});

            weights = xt::random::rand<float>({inputShape, outputShape}, 0, 1);
            drop = xt::empty<bool>({inputShape, outputShape});
        }

        ~outputLayer() = default;

        void forward(xt::xarray<float> input);

        void backward(
            xt::xarray<float> target,
            float tauxApprentissage);

};

#endif