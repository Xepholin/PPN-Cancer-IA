#ifndef CONVOLUTION_H
#define CONVOLUTION_H

#include <tuple>
#include <xtensor/xarray.hpp>

#include "activation.h"
#include "layer.h"
#include "relu.h"
#include "softmax.h"
#include "tools.h"

/**
 * @class Convolution
 * @brief Classe représentant une couche de convolution.
 */
class Convolution : public ILayer {
   public:
	int depth = 0;

	std::tuple<int, int, int> inputShape{0, 0, 0};					  // Tuple représentant la forme de l'entrée avec les dimensions (profondeur, hauteur, largeur).
	std::tuple<int, int, int> outputShape{0, 0, 0};					  // Tuple représentant la forme de sortie avec les dimensions (profondeur, hauteur, largeur).
	std::tuple<int, int, int, int, int> filtersShape{0, 0, 0, 0, 0};  // Tuple représentant la forme des filtres avec les dimensions (profondeur, hauteur, largeur, stride, padding).

	xt::xarray<float> filters;	// Les filtres de la couche de convolution.
	Activation *activation;		// L'activation à appliquer après la convolution.

	bool normalize = false;	 // Indique si la normalisation doit être appliquée après la convolution.

	float beta = 0.0;	// Le paramètre beta pour la normalisation.
	float gamma = 1.0;	// Le paramètre gamma pour la normalisation.

	/**
	 * @brief Constructeur de la classe Convolution.
	 *
	 * Initialise une couche de convolution avec les paramètres spécifiés.
	 *
	 * @param inputShape Tuple représentant la forme de l'entrée avec les dimensions (profondeur, hauteur, largeur).
	 * @param filtersShape Tuple représentant la forme des filtres avec les dimensions (profondeur, hauteur, largeur, stride, padding).
	 * @param activationType Le type d'activation à appliquer après la convolution (par défaut, pas d'activation).
	 * @param normalize Indique si la normalisation doit être appliquée après la convolution (par défaut, désactivée).
	 */
	Convolution(std::tuple<int, int, int> inputShape,
				std::tuple<int, int, int, int, int> filtersShape,
				ActivationType activationType = ActivationType::ACTIVATION_NO_TYPE,
				bool normalize = false) {
		name = "Convolution";

		this->depth = std::get<0>(inputShape);	// Nombre d'image dans la couche actuelle
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

		int outputHeight = ((inputHeight - filtersHeight + 2 * filtersPadding) / filtersStride) + 1;
		int outputWidth = ((inputWidth - filtersWidth + 2 * filtersPadding) / filtersStride) + 1;

		this->outputShape = std::tuple<int, int, int>(filtersDepth, outputHeight, outputWidth);
		this->input = xt::empty<float>({inputDepth, inputHeight, inputWidth});
		this->output = xt::empty<float>({filtersDepth, outputHeight, outputWidth});

		this->normalize = normalize;

		switch (activationType) {
			case ActivationType::ACTIVATION_NO_TYPE:
				this->filters = kernelsGaussianDistro(filtersDepth, depth, filtersHeight, filtersWidth);
				break;
			case relu: {
				this->activation = new ReLu(outputShape);
				this->heWeightsInit();
				break;
			}
			case softmax:
				perror("Convolution Activation Type Error");
				exit(0);
				break;
			case sigmoid:
				perror("Convolution Activation Type Error");
				exit(0);
				break;
			default:
				perror("Convolution Activation Type Error");
				exit(0);
		}
	}

	/**
	 * @brief Destructeur par défaut de la classe Convolution.
	 */
	~Convolution() = default;

	/**
	 * @brief Fonction de propagation  de la couche de convolution.
	 *
	 * Calcule la sortie de la couche de convolution pour une entrée donnée.
	 *
	 * @param input L'entrée de la couche de convolution.
	 * @param training Indique si le réseau est en phase d'entraînement (par défaut, true).
	 */
	void forward(xt::xarray<float> input, bool training = true) override;

	/**
	 * @brief Fonction de propagation arrière de la couche de convolution.
	 *
	 * Calcule le gradient de l'erreur par rapport à l'entrée de la couche de convolution.
	 *
	 * @param gradient Le gradient de l'erreur par rapport à la sortie de la couche suivante.
	 * @return Le gradient de l'erreur par rapport à l'entrée de la couche de convolution.
	 */
	xt::xarray<float> backward(xt::xarray<float> gradient) override;

	void print() const override;

	/**
	 * @brief Initialise les poids des filtres en utilisant l'initialisation de He.
	 */
	void heWeightsInit();

	/**
	 * @brief Initialise les poids des filtres en utilisant l'initialisation de Xavier-Glorot.
	 */
	void XGWeightsInit();
};

#endif	// CONVOLUTION_H
