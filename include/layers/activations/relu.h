#ifndef RELU_H
#define RELU_H

#include "activation.h"

class ReLu3D : public Activation {
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

            this->output = xt::empty<float>({depth, height, width});

        }

        void forward(xt::xarray<float> input) override;
        
        void backward(xt::xarray<float> gradient) override;

        void batchNorm();
};

class ReLu1D : public Activation {
      public:
        int depth = 0;

        // 1 x Longueur
        int inputShape = 0;
        int outputShape = 0;

        // Depth - Height 
        std::tuple<int, int> weightsShape{0, 0};

        // Depth - Height  
        xt::xarray<float> weights;

        float beta = 0;
        float gamma = 1;
    
        ReLu1D(int inputShape, int outputShape)   {

            this->inputShape = inputShape;
            this->outputShape = outputShape;
            this->weightsShape = std::tuple<int, int>{inputShape,outputShape};

            this->output = xt::empty<float>({outputShape});
            this->input = xt::empty<float>({inputShape});

            weights = xt::random::rand<float>({inputShape, outputShape}, 0, 1);   
        }

        ~ReLu1D() = default;

        void forward(xt::xarray<float> input) override;

        void backward(xt::xarray<float> gradient) override;
};


#endif