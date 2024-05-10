#ifndef SOFTMAX_H
#define SOFTMAX_H

#include <xtensor/xarray.hpp>

#include "activation.h"

/**
 * @class Softmax
 * @brief Classe représentant une fonction d'activation Softmax.
 */
class Softmax : public Activation {
   public:
	int inputShape = 0;
	int outputShape = 0;

	/**
	 * @brief Constructeur de la classe Softmax.
	 *
	 * Ce constructeur initialise une fonction d'activation Softmax avec les paramètres spécifiés.
	 *
	 * @param inputShape La taille de l'entrée de la fonction d'activation.
	 */
	Softmax(int inputShape) {
		name = "Softmax";

		this->inputShape = inputShape;
		this->outputShape = inputShape;

		this->input = xt::empty<float>({inputShape});
		this->output = xt::empty<float>({inputShape});
	}

	/**
	 * @brief Destructeur par défaut de la classe Softmax.
	 */
	~Softmax() = default;

	/**
	 * @brief Fonction de propagation avant de la fonction d'activation Softmax.
	 *
	 * Cette fonction calcule la sortie de la fonction d'activation Softmax pour une entrée donnée.
	 *
	 * @param input Entrée de la fonction d'activation Softmax.
	 * @param training Indique si le réseau est en phase d'entraînement (par défaut, true).
	 */
	virtual void forward(xt::xarray<float> input, bool training = true) override;

	/**
	 * @brief Fonction de propagation arrière de la fonction d'activation Softmax.
	 *
	 * Cette fonction calcule le gradient de l'erreur par rapport à l'entrée de la fonction Softmax.
	 *
	 * @param gradient Gradient de l'erreur par rapport à la sortie de la couche suivante.
	 * @return Gradient de l'erreur par rapport à l'entrée de la fonction Softmax.
	 */
	virtual xt::xarray<float> backward(xt::xarray<float> gradient) override;

	/**
	 * @brief Fonction calculant la dérivée de la fonction Softmax.
	 *
	 * Cette fonction calcule la dérivée de la fonction Softmax par rapport à un entrée.
	 *
	 * @param input Valeur d'entrée.
	 * @return Valeur de la dérivée de la fonction Softmax pour l'entrée donnée.
	 */
	virtual xt::xarray<float> prime(xt::xarray<float> input) override;

	void print() const override;
};

#endif	// SOFTMAX_H
