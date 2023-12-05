#ifndef NETWORK_H
#define NETWORK_H

#include <vector>
#include <tuple>

#include <xtensor/xarray.hpp>
#include <xtensor/xio.hpp>
#include <xtensor/xview.hpp>
#include <xtensor/xrandom.hpp>

// ILayer(xt::xarray<float> input, xt::xarray<float> output)
class ILayer {

    public:
        xt::xarray<float> input;
        xt::xarray<float> output;

        virtual void forward(xt::xarray<float> input);

        virtual void backward(xt::xarray<float> gradient);

        virtual float pooling(xt::xarray<float> matrix);

        virtual xt::xarray<float> poolingMatrice(xt::xarray<float> matrix);

        virtual float activation(xt::xarray<float> matrix);
};

// Pooling(int size, int stride, int padding, Pooling::PoolingType type)
struct Pooling  {
    
    enum PoolingType {
            NO_TYPE,
            MAX,    
            MIN,  
            AVG
    };

    int size = 1;
    int stride = 1;
    int padding = 0;
    PoolingType type = NO_TYPE;
};

// ConvolutionLayer(int depth, std::tuple<int, int, int> inputShape, std::tuple<int, int, int, int, int> filtersShape, Pooling pool)
class ConvolutionLayer : public ILayer   {
    
    public:
        int depth = 0;

        //Height - Width - Depth
        std::tuple<int, int, int> inputShape{0, 0, 0};
        std::tuple<int, int, int> outputShape{0, 0, 0};

        // Height - Width - Depth - Stride - Padding
        std::tuple<int, int, int, int, int> filtersShape{0, 0, 0, 0, 0};

        xt::xarray<float> filters;

        Pooling pool;

        int bias = 1;
        
        ConvolutionLayer(int depth, std::tuple<int, int, int> inputShape, std::tuple<int, int, int, int, int> filtersShape, Pooling pool)   {

            this->depth = depth;    // Nombre d'image dans la couche actuelle      
            this->inputShape = inputShape;
            this->filtersShape = filtersShape;
         
            int inputHeight = std::get<0>(inputShape);
            int inputWidth = std::get<1>(inputShape);
            int inputDepth = std::get<2>(inputShape);


            int filtersHeight = std::get<0>(filtersShape);
            int filtersWidth = std::get<1>(filtersShape);
            int filtersDepth = std::get<2>(filtersShape);

            this->outputShape = std::tuple<int, int, int>(inputHeight - filtersHeight + 1, inputWidth - filtersWidth + 1, filtersDepth * depth);

            this->pool = pool;

            filters = xt::random::rand<float>({depth, filtersDepth, filtersHeight, filtersWidth}, 0, 1);        
        }

        void forward(xt::xarray<float> input) override;

        void backward(xt::xarray<float> gradient) override;

        xt::xarray<float> poolingMatrice(xt::xarray<float> matrix) override;

        float pooling(xt::xarray<float> matrix) override;
        
        float activation(xt::xarray<float> matrix) override;

};

#endif