#ifndef CONVOLUTION_H
#define CONVOLUTION_H

#include <tuple>

#include "layer.h"
#include "activation.h"
#include "relu.h"
#include "softmax.h"
#include "tools.h"

class Convolution : public ILayer
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

        ActivationType activationType = ActivationType::ACTIVATION_NO_TYPE;
        Activation *activation;

		bool normalize = false;

        float beta = 0.0;
        float gamma = 1.0;

		/**
		 * @brief Constructeur de la classe Convolution.
		 *
		 * Ce constructeur initialise une couche de convolution avec les paramètres spécifiés.
		 *
		 * @param depth La profondeur de la couche de convolution.
		 * @param inputShape Tuple représentant la forme de l'entrée avec les dimensions (profondeur, hauteur, largeur).
		 * @param filtersShape Tuple représentant la forme des filtres avec les dimensions (profondeur, hauteur, largeur, stride, padding).
		 * @param activationType Le type d'activation à appliquer après la convolution (par défaut, pas d'activation).
		 * @param normalize Indique si la normalisation doit être appliquée après la convolution (par défaut, désactivée).
		*/
        Convolution(int depth, std::tuple<int, int, int> inputShape, 
                    std::tuple<int, int, int, int, int> filtersShape, 
                    ActivationType activationType = ActivationType::ACTIVATION_NO_TYPE,
					bool normalize = false)
        {
			name = "Convolution";

            this->depth = depth; // Nombre d'image dans la couche actuelle
            this->inputShape = inputShape;
            this->filtersShape = filtersShape;
            this->activationType = activationType;

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
			this->activationType = activationType;
			this->normalize = normalize;

            switch (this->activationType)
            {
                case ActivationType::ACTIVATION_NO_TYPE:
                    this->filters = kernelsGaussianDistro(filtersDepth, depth, filtersHeight, filtersWidth);
                    break;
                case ActivationType::ACTIVATION_RELU:   {
                    this->activation = new ReLu(outputShape);
                    this->heWeightsInit();
                    break;
                }
                case ActivationType::ACTIVATION_SOFTMAX:
                    perror("Convolution Activation Type Error");
                    break;
                default:
                    perror("Convolution Activation Type Error");
            }
        }

        ~Convolution()  {
            delete this->activation;
        }

        void forward(xt::xarray<float> input) override;

        xt::xarray<float> backward(xt::xarray<float> gradient, float learningRate) override;

        void print() const override;

        void heWeightsInit();

        void XGWeightsInit();
};

#endif