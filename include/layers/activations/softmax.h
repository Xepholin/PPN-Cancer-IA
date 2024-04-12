#ifndef SOFTMAX_H
#define SOFTMAX_H

#include <tuple>

#include "activation.h"

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

	~Softmax() = default;

	virtual void forward(xt::xarray<float> input, bool training = true) override;

	virtual xt::xarray<float> backward(xt::xarray<float> gradient, float learningRate) override;

	virtual xt::xarray<float> prime(xt::xarray<float> input) override;

	void layerGradient(xt::xarray<float> trueLabel);

	void print() const override;

	xt::xarray<float> softmaxJacobien();

	xt::xarray<float> softmaxGradient();
};

#endif