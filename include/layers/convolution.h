#ifndef CONVOLUTION_H
#define CONVOLUTION_H

#include <tuple>

#include <xtensor/xarray.hpp>
#include <xtensor/xrandom.hpp>

#include "tools.h"

// Convolution(int depth, std::tuple<int, int, int> inputShape, std::tuple<int, int, int, int, int> filtersShape)
class Convolution
{

    public:
        xt::xarray<float> input;
        xt::xarray<float> output;

        int depth = 0;

        // Depth - Height - Width
        std::tuple<int, int, int> inputShape{0, 0, 0};
        std::tuple<int, int, int> outputShape{0, 0, 0};

        // Depth - Height - Width - Stride - Padding
        std::tuple<int, int, int, int, int> filtersShape{0, 0, 0, 0, 0};

        // nbFilter - Depth - Height - Width -
        xt::xarray<float> filters;

        int bias = 1;

        float beta = 0.0;
        float gamma = 1.0;

        Convolution(int depth, std::tuple<int, int, int> inputShape, std::tuple<int, int, int, int, int> filtersShape)
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

            // filters = kernelsGaussianDistro(filtersDepth, depth, filtersHeight, filtersWidth);

            filters = xt::random::randn<float>({filtersDepth, depth, filtersHeight, filtersWidth});
        }

        ~Convolution() = default;

        void forward(xt::xarray<float> input);

        void backward(xt::xarray<float> gradient);
};

#endif