#ifndef NETWORK_H
#define NETWORK_H

#include <vector>
#include <tuple>

#include <xtensor/xarray.hpp>
#include <xtensor/xio.hpp>
#include <xtensor/xview.hpp>
#include <xtensor/xrandom.hpp>

#include "convolution.h"
#include "tools.h"

// ILayer(xt::xarray<float> input, xt::xarray<float> output)
class ILayer
{

public:
    xt::xarray<float> input;
    xt::xarray<float> output;

    virtual void forward(xt::xarray<float> input);

    virtual void backward(xt::xarray<float> gradient);
};

// ConvolutionLayer(int depth, std::tuple<int, int, int> inputShape, std::tuple<int, int, int, int, int> filtersShape)
class ConvolutionLayer : public ILayer
{

public:
    int depth = 0;

    // Depth - Height - Width
    std::tuple<int, int, int> inputShape{0, 0, 0};
    std::tuple<int, int, int> outputShape{0, 0, 0};

    // Depth - Height - Width - Stride - Padding
    std::tuple<int, int, int, int, int> filtersShape{0, 0, 0, 0, 0};

    // nbFilter - Depth - Height - Width -
    xt::xarray<float> filters;

    int bias = 1;

    ConvolutionLayer(int depth, std::tuple<int, int, int> inputShape, std::tuple<int, int, int, int, int> filtersShape)
    {

        this->depth = depth; // Nombre d'image dans la couche actuelle
        this->inputShape = inputShape;
        this->filtersShape = filtersShape;

        int inputDepth = std::get<0>(inputShape);
        int inputHeight = std::get<1>(inputShape);
        int inputWidth = std::get<2>(inputShape);

        int filtersDepth = std::get<0>(filtersShape);
        int filtersHeight = std::get<1>(filtersShape);
        int filtersWidth = std::get<2>(filtersShape);
        int filtersStride = std::get<3>(filtersShape);
        int filtersPadding = std::get<4>(filtersShape);

        int outputHeight = (inputHeight - filtersHeight + 2 * filtersPadding) / filtersStride + 1;
        int outputWidth = (inputWidth - filtersWidth + 2 * filtersPadding) / filtersStride + 1;

        this->outputShape = std::tuple<int, int, int>(filtersDepth, outputHeight, outputWidth);
        this->input = xt::empty<float>({inputDepth, inputHeight, inputWidth});
        this->output = xt::empty<float>({filtersDepth, outputHeight, outputWidth});

        filters = kernelsGaussianDistro(filtersDepth, depth, filtersHeight, filtersWidth);
    }

    ~ConvolutionLayer() = default;

    void forward(xt::xarray<float> input) override;

    void backward(xt::xarray<float> gradient) override;
};

// Pooling(std::tuple<int, int, int> inputShape, int size, int stride, int padding, Pooling::PoolingType type)
class PoolingLayer : public ILayer
{

public:
    enum PoolingType
    {
        NO_TYPE,
        MAX,
        MIN,
        AVG
    };

    std::tuple<int, int, int> inputShape{0, 0, 0};
    std::tuple<int, int, int> outputShape{0, 0, 0};

    int size = 1;
    int stride = 1;
    int padding = 0;
    PoolingType type = NO_TYPE;

    PoolingLayer(std::tuple<int, int, int> inputShape, int size, int stride, int padding, PoolingType type)
    {
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
        this->type = type;
    }

    ~PoolingLayer() = default;

    void forward(xt::xarray<float> input) override;

    void backward(xt::xarray<float> gradient) override;

    float pooling(xt::xarray<float> matrix);

    xt::xarray<float> poolingMatrice(xt::xarray<float> matrix);
};

// Activation()
class ActivationLayer : public ILayer
{

public:
    void forward(xt::xarray<float> input) override;

    void backward(xt::xarray<float> gradient) override;

    xt::xarray<float> activation(xt::xarray<float> matrix);
};

// DenseLayer(int depth, std::tuple<int, int, int> inputShape, std::tuple<int, int, int, int, int> weightsShape)
class DenseLayer : public ILayer
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

    DenseLayer(int inputShape, int outputShape)
    {

        this->inputShape = inputShape;
        this->outputShape = outputShape;
        this->weightsShape = std::tuple<int, int>{inputShape, outputShape};

        this->output = xt::empty<float>({outputShape});
        this->input = xt::empty<float>({inputShape});

        weights = xt::random::rand<float>({inputShape, outputShape}, 0, 1);
    }

    ~DenseLayer() = default;

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

xt::xarray<float> flatten(xt::xarray<float> input);

#endif
