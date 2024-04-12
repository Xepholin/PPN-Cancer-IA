#ifndef DENSE_H
#define DENSE_H

#include "layer.h"
#include "activation.h"
#include "relu.h"
#include "softmax.h"
#include "sigmoid.h"
#include "tools.h"
#include "loss.h"

class Dense : public ILayer
{
    public:
        // 1 x Longueur
        int inputShape = 0;
        int outputShape = 0;

        // Height -Width
        std::tuple<int, int> weightsShape{0, 0};

        // Height -Width
        xt::xarray<float> weights;
        xt::xarray<float> weightsGradient;

		int dropRate = 0;
        xt::xarray<bool> drop;

        xt::xarray<float> bias;
        xt::xarray<float> biasGradient;

        xt::xarray<float> gammas;
        xt::xarray<float> gammasGradient;

        xt::xarray<float> betas;
        xt::xarray<float> betasGradient;

        Activation *activation;

		bool normalize;

        bool flatten;

		xt::xarray<float> baOutput;
		xt::xarray<float> bnOutput;

		Loss lossType;

		/**
		 * @brief Constructeur de la classe Dense.
		 *
		 * Ce constructeur initialise une couche dense avec les paramètres spécifiés.
		 *
		 * @param inputShape La taille de l'entrée de la couche dense.
		 * @param outputShape La taille de la sortie de la couche dense.
		 * @param activationType Le type d'activation à appliquer après la couche dense (par défaut, pas d'activation).
		 * @param normalize Indique si la normalisation doit être appliquée après la couche dense (par défaut, désactivée).
		 * @param flatten Indique si l'entrée doit être aplatie (par défaut, désactivé).
		*/
        Dense(int inputShape, int outputShape,
			  ActivationType activationType = ActivationType::ACTIVATION_NO_TYPE,
			  int dropRate = 0, bool normalize = false, bool flatten = false)	{
			name = "Dense";
            this->inputShape = inputShape;
            this->outputShape = outputShape;
            this->weightsShape = std::tuple<int, int>{inputShape,outputShape};

            this->output = xt::empty<float>({outputShape});
            this->input = xt::empty<float>({inputShape});
			this->baOutput = xt::empty<float>({outputShape});
			this->bnOutput = xt::empty<float>({outputShape});

			this->weightsGradient = xt::zeros<float>({inputShape,outputShape});

			this->bias = xt::random::randn<float>({outputShape});
			this->biasGradient = xt::zeros<float>({outputShape});

			this->gammas = xt::ones<float>({outputShape});
			this->gammasGradient = xt::zeros<float>({outputShape});

			this->betas = xt::zeros<float>({outputShape});
			this->betasGradient = xt::zeros<float>({outputShape});
			
			this->dropRate = dropRate;
            drop = xt::zeros<bool>({inputShape});

			this->normalize = normalize;

            this->flatten = flatten;

            switch (activationType)
            {
                case ActivationType::ACTIVATION_NO_TYPE:
                    this->activation = new Activation;
                    this->weights = xt::random::randn<float>({inputShape,outputShape}, 0, 1.0/inputShape);
                    break;

                case relu:
                    this->activation = new ReLu(std::tuple<int, int ,int>{1, 1, outputShape});
					// this->weights = xt::random::randn<float>({outputShape, inputShape}, 0, 1.0/inputShape);
                    this->heWeightsInit();
                    break;

                case softmax:
                    this->activation = new Softmax(outputShape);
					// this->weights = xt::random::randn<float>({outputShape, inputShape}, 0, 1.0/inputShape);
                    this->XGWeightsInit();
                    break;
                    
                case sigmoid:
                    this->activation = new Sigmoid(outputShape);
					// this->weights = xt::random::randn<float>({outputShape, inputShape}, 0, 1.0/inputShape);
                    this->XGWeightsInit();
                    break;
                    
                default:
                    perror("Dense Activation Type Error");
					exit(0);
            }
        }

        ~Dense() = default;

        virtual void forward(xt::xarray<float> input, bool training = true) override;

        virtual xt::xarray<float> backward(xt::xarray<float> gradient);
                
        xt::xarray<float> oldbackward(
				xt::xarray<float> gradient);

		void norm();

        void print() const override;

        void dropout();

        void printDropout(uint16_t dropRate) const;

        void heWeightsInit();

        void XGWeightsInit();
};

#endif