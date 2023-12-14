#ifndef POOLING_H
#define POOLING_H

#include <tuple>

#include "layer.h"

enum PoolingType
{
    POOLING_NO_TYPE,
    POOLING_MAX,
    POOLING_MIN,
    POOLING_AVG
};

std::ostream& operator<<(std::ostream& out, const PoolingType value);

// Pooling(std::tuple<int, int, int> inputShape, int size, int stride, int padding, Pooling::PoolingType type)
class Pooling : public ILayer
{

    public:

        std::tuple<int, int, int> inputShape{0, 0, 0};
        std::tuple<int, int, int> outputShape{0, 0, 0};

        int size = 1;
        int stride = 1;
        int padding = 0;
        PoolingType type = PoolingType::POOLING_NO_TYPE;

        Pooling(std::tuple<int, int, int> inputShape,
                int size, int stride, int padding, 
                PoolingType poolingType = PoolingType::POOLING_NO_TYPE)
        {   
            this->name = "Pooling";
            this->inputShape = inputShape;

            int depth = std::get<0>(inputShape);
            int height = std::get<1>(inputShape);
            int width = std::get<2>(inputShape);

            int outputHeight = (height - size + 2 * padding) / stride + 1;
            int outputWidth = (width - size + 2 * padding) / stride + 1;

            this->outputShape = std::tuple<int, int, int>(depth, outputHeight, outputWidth);

            this->output = xt::empty<float>({depth, outputHeight, outputWidth});

            this->size = size;
            this->stride = stride;
            this->padding = padding;
            this->type = poolingType;
        }

        ~Pooling() = default;

        void forward(xt::xarray<float> input) override;

        void backward(xt::xarray<float> cost, float learningRate) override;

        void print() const override;

        float pooling(xt::xarray<float> matrix);

        xt::xarray<float> poolingMatrice(xt::xarray<float> matrix);
};

#endif