#ifndef DENSE_H
#define DENSE_H

#include <tuple>
#include <xtensor/xarray.hpp>

#include "activation.h"
#include "layer.h"
#include "loss.h"
#include "relu.h"
#include "sigmoid.h"
#include "softmax.h"
#include "tools.h"

/**
 * @class Dense
 * @brief Classe représentant une couche dense.
 */
class Dense : public ILayer {
   public:
	int inputShape = 0;	  // La taille de l'entrée
	int outputShape = 0;  // La taille de la sortie

	std::tuple<int, int> weightsShape{0, 0};  // Les dimensions des poids de la couche dense

	xt::xarray<float> weights;			// Les poids de la couche dense
	xt::xarray<float> weightsGradient;	// Les gradients des poids de la couche dense

	int dropRate = 0;		// Le taux de dropout
	xt::xarray<bool> drop;	// Les masques de dropout

	xt::xarray<float> bias;			 // Les biais de la couche dense
	xt::xarray<float> biasGradient;	 // Les gradients des biais de la couche dense

	xt::xarray<float> gammas;		   // Les paramètres gamma pour la normalisation
	xt::xarray<float> gammasGradient;  // Les gradients des paramètres gamma pour la normalisation

	xt::xarray<float> betas;		  // Les paramètres beta pour la normalisation
	xt::xarray<float> betasGradient;  // Les gradients des paramètres beta pour la normalisation

	Activation *activation;	 // L'activation à appliquer après la couche dense

	bool normalize;	 // Indique si la normalisation doit être appliquée après la couche dense

	bool flatten;  // Indique si l'entrée doit être aplatie avant d'être passée à la couche dense

	xt::xarray<float> baOutput;	 // La sortie avant l'activation
	xt::xarray<float> bnOutput;	 // La sortie après la normalisation

	Loss lossType;	// Le type de fonction de perte à utiliser pour l'entraînement

	/**
	 * @brief Constructeur de la classe Dense.
	 *
	 * Initialise une couche dense avec les paramètres spécifiés.
	 *
	 * @param inputShape La taille de l'entrée.
	 * @param outputShape La taille de la sortie.
	 * @param activationType Le type d'activation à appliquer après la couche dense (par défaut, pas d'activation).
	 * @param dropRate Le taux de dropout à appliquer (par défaut, 0).
	 * @param normalize Indique si la normalisation doit être appliquée après la couche dense (par défaut, désactivée).
	 * @param flatten Indique si l'entrée doit être aplatie avant d'être passée à la couche dense (par défaut, désactivée).
	 */
	Dense(int inputShape, int outputShape,
		  ActivationType activationType = ActivationType::ACTIVATION_NO_TYPE,
		  int dropRate = 0, bool normalize = false, bool flatten = false) {
		name = "Dense";
		this->inputShape = inputShape;
		this->outputShape = outputShape;
		this->weightsShape = std::tuple<int, int>{inputShape, outputShape};

		this->output = xt::empty<float>({outputShape});
		this->input = xt::empty<float>({inputShape});
		this->baOutput = xt::empty<float>({outputShape});
		this->bnOutput = xt::empty<float>({outputShape});

		this->weightsGradient = xt::zeros<float>({inputShape, outputShape});

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

		switch (activationType) {
			case ActivationType::ACTIVATION_NO_TYPE:
				this->activation = new Activation;
				this->weights = xt::random::randn<float>({inputShape, outputShape}, 0, 1.0 / inputShape);
				break;

			case relu:
				this->activation = new ReLu(std::tuple<int, int, int>{1, 1, outputShape});
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

	/**
	 * @brief Destructeur par défaut de la classe Dense.
	 */
	~Dense() = default;

	/**
	 * @brief Fonction de propagation avant de la couche dense.
	 *
	 * Calcule la sortie de la couche dense pour une entrée donnée.
	 *
	 * @param input L'entrée de la couche dense.
	 * @param training Indique si le réseau est en phase d'entraînement (par défaut, true).
	 */
	void forward(xt::xarray<float> input, bool training = true) override;

	/**
	 * @brief Fonction de propagation arrière de la couche dense.
	 *
	 * Calcule le gradient de l'erreur par rapport à l'entrée de la couche dense.
	 *
	 * @param gradient Le gradient de l'erreur par rapport à la sortie de la couche suivante.
	 * @return Le gradient de l'erreur par rapport à l'entrée de la couche dense.
	 */
	xt::xarray<float> backward(xt::xarray<float> gradient);

	void print() const override;

	void printDropout(uint16_t dropRate) const;

	/**
	 * @brief Applique la normalisation sur la sortie de la couche dense.
	 */
	void norm();

	/**
	 * @brief Applique la technique de dropout sur l'entrée de la couche dense.
	 */
	void dropout();

	/**
	 * @brief Initialise les poids des filtres en utilisant l'initialisation de He.
	 */
	void heWeightsInit();

	/**
	 * @brief Initialise les poids des filtres en utilisant l'initialisation de Xavier/Glorot.
	 */
	void XGWeightsInit();
};

#endif	// DENSE_H
