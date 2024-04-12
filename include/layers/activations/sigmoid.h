#ifndef SIGMOID_H
#define SIGMOID_H

#include <tuple>

#include "activation.h"

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

	~Sigmoid() = default;

	virtual void forward(xt::xarray<float> input, bool training = true) override;

	virtual xt::xarray<float> backward(xt::xarray<float> gradient) override;

	virtual xt::xarray<float> prime(xt::xarray<float> input) override;

	void layerGradient(xt::xarray<float> trueLabel);

	void print() const override;

	xt::xarray<float> softmaxJacobien();

	xt::xarray<float> softmaxGradient();
};

#endif