#ifndef SIGMOID_H
#define SIGMOID_H

#include <xtensor/xarray.hpp>

#include "activation.h"

/**
 * @class Sigmoid
 * @brief Classe représentant une fonction d'activation Sigmoid.
 */
class Sigmoid : public Activation {
   public:
	int inputShape = 0;
	int outputShape = 0;

	/**
	 * @brief Constructeur de la classe Sigmoid.
	 *
	 * Ce constructeur initialise une fonction d'activation Sigmoid avec les paramètres spécifiés.
	 *
	 * @param inputShape La taille de l'entrée de la fonction d'activation.
	 */
	Sigmoid(int inputShape) {
		name = "Sigmoid";

		this->inputShape = inputShape;
		this->outputShape = inputShape;

		this->input = xt::empty<float>({inputShape});
		this->output = xt::empty<float>({inputShape});
	}

	/**
	 * @brief Destructeur par défaut de la classe Sigmoid.
	 */
	~Sigmoid() = default;

	/**
	 * @brief Fonction propagation avant de la fonction d'activation Sigmoid.
	 *
	 * Cette fonction calcule la sortie de la fonction d'activation Sigmoid pour une entrée donnée.
	 *
	 * @param input Entrée de la fonction d'activation Sigmoid.
	 * @param training Indique si le réseau est en phase d'entraînement (par défaut, true).
	 */
	virtual void forward(xt::xarray<float> input, bool training = true) override;

	/**
	 * @brief Fonction de propagation arrière de la fonction d'activation Sigmoid.
	 *
	 * Cette fonction calcule le gradient de l'erreur par rapport à l'entrée de la fonction Sigmoid.
	 *
	 * @param gradient Gradient de l'erreur par rapport à la sortie de la couche suivante.
	 * @return Gradient de l'erreur par rapport à l'entrée de la fonction Sigmoid.
	 */
	virtual xt::xarray<float> backward(xt::xarray<float> gradient) override;

	/**
	 * @brief Fonction calculant la dérivée de la fonction Sigmoid.
	 *
	 * Cette fonction calcule la dérivée de la fonction Sigmoid par rapport à un entrée.
	 *
	 * @param input Valeur d'entrée.
	 * @return Valeur de la dérivée de la fonction Sigmoid pour l'entrée donnée.
	 */
	virtual xt::xarray<float> prime(xt::xarray<float> input) override;

	void print() const override;
};

#endif