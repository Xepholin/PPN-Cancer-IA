#ifndef ACTIVATIONS_H
#define ACTIVATIONS_H

#include "network.h"

class ReLu : public ActivationLayer {
    public:
        std::tuple<int, int, int> inputShape{0, 0, 0};
        std::tuple<int, int, int> outputShape{0, 0, 0};

        ReLu(std::tuple<int, int, int> inputShape)  {
            this->inputShape = inputShape;
            this->outputShape = inputShape;

            int depth = std::get<0>(inputShape);
            int height = std::get<1>(inputShape);
            int width = std::get<2>(inputShape);

            this->output = xt::empty<float>({depth, height, width});
        }

        void forward(xt::xarray<float> input) override;
        
        void backward(xt::xarray<float> gradient) override;

        xt::xarray<float> activation(xt::xarray<float> matrix) override;
};

#endif