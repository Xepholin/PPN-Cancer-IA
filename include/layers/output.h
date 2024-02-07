#ifndef OUTPUTLAYER_H
#define OUTPUTLAYER_H

#include "activation.h"
#include "layer.h"
#include "relu.h"
#include "softmax.h"
#include "tools.h"

class Output : public ILayer {
   public:
	// 1 x Longueur
	int inputShape = 0;
	int outputShape = 0;

	// Height -Width
	std::tuple<int, int> weightsShape{0, 0};

	// Height -Width
	xt::xarray<float> weights;

	xt::xarray<bool> drop;

	xt::xarray<float> bias;

	ActivationType activationType = ActivationType::ACTIVATION_NO_TYPE;
	Activation *activation;

	bool normalize = false;

	xt::xarray<float> bOutput;

	/**
	 * @brief Constructeur de la classe Output.
	 *
	 * Ce constructeur initialise une couche de sortie avec les paramètres spécifiés.
	 *
	 * @param inputShape La taille de l'entrée de la couche de sortie.
	 * @param outputShape La taille de la sortie de la couche de sortie.
	 * @param activationType Le type d'activation à appliquer après la couche de sortie (par défaut, pas d'activation).
	 * @param normalize Indique si la normalisation doit être appliquée après la couche de sortie (par défaut, désactivée).
	*/
	Output(int inputShape, int outputShape,
		   ActivationType activationType = ActivationType::ACTIVATION_NO_TYPE,
		   bool normalize = false) {
		name = "Output";

		this->inputShape = inputShape;
		this->outputShape = outputShape;
		this->weightsShape = std::tuple<int, int>{inputShape, outputShape};

		this->input = xt::empty<float>({inputShape});
		this->output = xt::empty<float>({outputShape});
		this->bOutput = xt::empty<float>({outputShape});
		this->bias = xt::random::randn<float>({outputShape});

		drop = xt::zeros<bool>({inputShape});

		this->activationType = activationType;

		this->normalize = normalize;

		switch (this->activationType) {
			case ActivationType::ACTIVATION_NO_TYPE:
				this->activation = new Activation;
				this->weights = xt::random::randn<float>({inputShape, outputShape});
				break;

			case ActivationType::ACTIVATION_RELU:
				this->activation = new ReLu(std::tuple<int, int, int>{1, 1, outputShape});
				this->weights = xt::random::randn<float>({inputShape, outputShape});
				// this->heWeightsInit();
				break;

			case ActivationType::ACTIVATION_SOFTMAX:
				this->activation = new Softmax(outputShape);
				this->weights = xt::random::randn<float>({inputShape, outputShape});
				// this->XGWeightsInit();
				break;

			default:
				perror("Dense Activation Type Error");
		}
	}

	~Output() {
		delete this->activation;
	}

	virtual void forward(xt::xarray<float> input) override;

	virtual xt::xarray<float> backward(
		xt::xarray<float> label,
    	float learningRate);

	void print() const override;

	void heWeightsInit();

	void XGWeightsInit();
};

#endif