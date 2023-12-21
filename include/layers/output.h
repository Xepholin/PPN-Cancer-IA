#ifndef OUTPUTLAYER_H
#define OUTPUTLAYER_H

#include "layer.h"

// Output(int depth, std::tuple<int, int, int> inputShape, std::tuple<int, int, int, int, int> weightsShape)
class Output : public ILayer
{
    public:

        int depth = 0;

        // 1 x Longueur
        int inputShape = 0;
        int outputShape = 0;

        // Height -Width
        std::tuple<int, int> weightsShape{0, 0};

        int bias = 1;

        Output(int inputShape, int outputShape)
        {
            this->name = "Output";
            this->inputShape = inputShape;
            this->outputShape = outputShape;
            this->weightsShape = std::tuple<int, int>{inputShape, outputShape};

            this->output = xt::empty<float>({outputShape});
            this->input = xt::empty<float>({inputShape});
        }

        ~Output() = default;

        void forward(xt::xarray<float> input) override;

        void backward(
            xt::xarray<float> cost,
            float learningRate) override;

        void print() const override;

};

#endif