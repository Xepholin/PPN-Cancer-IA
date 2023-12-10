#ifndef RELU_H
#define RELU_H

#include <tuple>

#include "activation.h"

class ReLu3D : public Activation
{
    public:
        std::tuple<int, int, int> inputShape{0, 0, 0};
        std::tuple<int, int, int> outputShape{0, 0, 0};
        
        float beta = 0;
        float gamma = 1;
        
        ReLu3D(std::tuple<int, int, int> inputShape)  {
            this->inputShape = inputShape;
            this->outputShape = inputShape;

            int depth = std::get<0>(inputShape);
            int height = std::get<1>(inputShape);
            int width = std::get<2>(inputShape);

            this->input = xt::empty<float>({depth, height, width});
            this->output = xt::empty<float>({depth, height, width});

        }

        void forward(xt::xarray<float> input) override;
        
        void backward() override;

        void batchNorm();
        
};

class ReLu1D : public Activation {
      public:
        int depth = 0;

        // 1 x Longueur
        int inputShape = 0;
        int outputShape = 0;

        float beta = 0;
        float gamma = 1;
    
        ReLu1D(int inputShape)   {

            this->inputShape = inputShape;
            this->outputShape = inputShape;
            
            this->input = xt::empty<float>({inputShape});
            this->output = xt::empty<float>({outputShape});
        }

        ~ReLu1D() = default;

        void forward(xt::xarray<float> input) override;

        void backward() override;
};


#endif