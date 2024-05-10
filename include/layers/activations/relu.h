#ifndef RELU_H
#define RELU_H

#include <tuple>
#include <xtensor/xarray.hpp>

#include "activation.h"

/**
 * @class ReLu
 * @brief Classe représentant une fonction d'activation ReLu.
 */
class ReLu : public Activation {
   public:
	std::tuple<int, int, int> inputShape{0, 0, 0};	 // Forme de l'entrée de la couche ReLu.
	std::tuple<int, int, int> outputShape{0, 0, 0};	 // Forme de la sortie de la couche ReLu.

	/**
	 * @brief Constructeur de la classe ReLu.
	 *
	 * Ce constructeur initialise une fonction d'activation ReLu avec les paramètres spécifiés.
	 *
	 * @param inputShape Tuple représentant la forme de l'entrée avec les dimensions (profondeur, hauteur, largeur).
	 */
	ReLu(std::tuple<int, int, int> inputShape) {
		name = "ReLu";

		this->inputShape = inputShape;
		this->outputShape = inputShape;

		int depth = std::get<0>(inputShape);
		int height = std::get<1>(inputShape);
		int width = std::get<2>(inputShape);

		this->input = xt::empty<float>({depth, height, width});
		this->output = xt::empty<float>({depth, height, width});
	}

	/**
	 * @brief Destructeur par défaut de la classe ReLu.
	 */
	~ReLu() = default;

	/**
	 * @brief Fonction de propagation avant de la couche ReLu.
	 *
	 * Cette fonction calcule la sortie de la couche ReLu pour une entrée donnée.
	 *
	 * @param input Entrée de la couche ReLu.
	 * @param training Indique si le réseau est en phase d'entraînement (par défaut, true).
	 */
	virtual void forward(xt::xarray<float> input, bool training = true) override;

	/**
	 * @brief Fonction de propagation arrière de la couche ReLu.
	 *
	 * Cette fonction calcule le gradient de l'erreur par rapport à l'entrée de la couche ReLu.
	 *
	 * @param gradient Gradient de l'erreur par rapport à la sortie de la couche suivante.
	 * @return Gradient de l'erreur par rapport à l'entrée de la couche ReLu.
	 */
	virtual xt::xarray<float> backward(xt::xarray<float> gradient) override;

	/**
	 * @brief Fonction calculant la dérivée de la fonction ReLu.
	 *
	 * Cette fonction calcule la dérivée de la fonction ReLu par rapport à une entrée.
	 *
	 * @param input Valeur d'entrée.
	 * @return Valeur de la dérivée de la fonction ReLu pour l'entrée donnée.
	 */
	virtual xt::xarray<float> prime(xt::xarray<float> input) override;

	void print() const override;
};

#endif
