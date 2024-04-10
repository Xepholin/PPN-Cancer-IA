#ifndef RELU_H
#define RELU_H

#include <tuple>

#include "activation.h"

class ReLu : public Activation
{
    public:
        std::tuple<int, int, int> inputShape{0, 0, 0};
        std::tuple<int, int, int> outputShape{0, 0, 0};

		/**
		 * @brief Constructeur de la classe ReLu.
		 *
		 * Ce constructeur initialise une fonction d'activation ReLu avec les paramètres spécifiés.
		 *
		 * @param inputShape Tuple représentant la forme de l'entrée avec les dimensions (profondeur, hauteur, largeur).
		*/
        ReLu(std::tuple<int, int, int> inputShape)
        {
			name = "ReLu";
			
            this->inputShape = inputShape;
            this->outputShape = inputShape;

            int depth = std::get<0>(inputShape);
            int height = std::get<1>(inputShape);
            int width = std::get<2>(inputShape);

            this->input = xt::empty<float>({depth, height, width});
            this->output = xt::empty<float>({depth, height, width});
        }

        ~ReLu() = default;

        virtual void forward(xt::xarray<float> input) override;
        
        virtual xt::xarray<float> backward(xt::xarray<float> gradient, float learningRate) override;

        virtual xt::xarray<float> prime(xt::xarray<float> input) override;

        void print() const override;
        
};

#endif