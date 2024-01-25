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

class Pooling : public ILayer
{

    public:
        std::tuple<int, int, int> inputShape{0, 0, 0};
        std::tuple<int, int, int> outputShape{0, 0, 0};

        int size = 1;
        int stride = 1;
        int padding = 0;
        PoolingType type = PoolingType::POOLING_NO_TYPE;

		/**
		 * @brief Constructeur de la classe Pooling.
		 *
		 * Ce constructeur initialise une couche de pooling avec les paramètres spécifiés.
		 *
		 * @param inputShape Tuple représentant la forme de l'entrée avec les dimensions (profondeur, hauteur, largeur).
		 * @param size La taille du filtre de pooling.
		 * @param stride Le décalage entre les filtres lors de l'application du pooling.
		 * @param padding Le rembourrage à appliquer autour de l'entrée avant le pooling.
		 * @param poolingType Le type de pooling à appliquer (par défaut, pas de pooling).
		*/
        Pooling(std::tuple<int, int, int> inputShape,
                int size, int stride, int padding, 
                PoolingType poolingType = PoolingType::POOLING_NO_TYPE)
        {
			name = "Pooling";
			
            this->inputShape = inputShape;

            int depth = std::get<0>(inputShape);
            int height = std::get<1>(inputShape);
            int width = std::get<2>(inputShape);

            int outputHeight = (height - size + 2 * padding) / stride + 1;
            int outputWidth = (width - size + 2 * padding) / stride + 1;

            this->outputShape = std::tuple<int, int, int>(depth, outputHeight, outputWidth);

            this->input = xt::empty<float>({depth, height, width});
            this->output = xt::empty<float>({depth, outputHeight, outputWidth});

            this->size = size;
            this->stride = stride;
            this->padding = padding;
            this->type = poolingType;
        }

        ~Pooling() = default;

        void forward(xt::xarray<float> input) override;

        xt::xarray<float> backward(xt::xarray<float> gradient, float learningRate) override;

        void print() const override;

        float pooling(xt::xarray<float> matrix);

        xt::xarray<float> poolingMatrice(xt::xarray<float> matrix);
};

#endif