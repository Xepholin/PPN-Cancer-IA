#ifndef NETWORK_H
#define NETWORK_H

#include <xtensor/xarray.hpp>
#include <xtensor/xio.hpp>
#include <xtensor/xview.hpp>
#include <xtensor/xrandom.hpp>

class ILayer {
    public:
        xt::xarray<float> input;
        xt::xarray<float> output;

        virtual void forward(xt::xarray<float> input);

        virtual void backward(xt::xarray<float> gradient);
};

class Convolution : public ILayer   {
    public:
        int depth = 0;

        int inputHeight = 0;
        int inputWidth = 0;
        int inputDepth = 0;

        int outputHeight = 0;
        int outputWidth = 0;
        int outputDepth = 0;

        int kernelHeight = 0;
        int kernelWidth = 0;

        xt::xarray<float> biases;
        xt::xarray<float> kernels;

        Convolution(int inputHeight, int inputWidth, int inputDepth, int kernelSize, int depth)   {
            this->depth = depth;
            this->inputHeight = inputHeight;
            this->inputWidth = inputWidth;
            this->inputDepth = inputDepth;

            this->outputHeight = inputHeight - kernelSize + 1;
            this->outputWidth = inputWidth - kernelSize + 1;
            this->outputDepth = depth;

            this->kernelHeight = kernelSize;
            this->kernelWidth = kernelSize;

            biases = xt::ones<int>({kernelHeight, kernelWidth});
            kernels = xt::random::randn<float>({depth, kernelHeight, kernelWidth});
        }

        void forward(xt::xarray<float> input) override;

        virtual void backward(xt::xarray<float> gradient) override;

};

#endif