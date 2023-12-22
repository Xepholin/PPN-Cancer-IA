#ifndef OUTPUTLAYER_H
#define OUTPUTLAYER_H

#include "activation.h"
#include "layer.h"
#include "relu.h"
#include "softmax.h"
#include "tools.h"

// Output(int inputShape, int outputShape)
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

	int bias = 1;

	ActivationType activationType = ActivationType::ACTIVATION_NO_TYPE;
	Activation *activation;

	bool normalize = false;

	bool flatten = false;

	xt::xarray<float> bOutput;

	Output(int inputShape, int outputShape,
		   ActivationType activationType = ActivationType::ACTIVATION_NO_TYPE,
		   bool normalize = false, bool flatten = false) {
		this->name = "Output";
		this->inputShape = inputShape;
		this->outputShape = outputShape;
		this->weightsShape = std::tuple<int, int>{inputShape, outputShape};

		this->input = xt::empty<float>({inputShape});
		this->output = xt::empty<float>({outputShape});
		this->bOutput = xt::empty<float>({outputShape});

		drop = xt::zeros<bool>({inputShape});

		this->activationType = activationType;

		this->normalize = normalize;

		this->flatten = flatten;

		switch (this->activationType) {
			case ActivationType::ACTIVATION_NO_TYPE:
				this->activation = new Activation;
				this->weights = xt::random::randn<float>({inputShape, outputShape});
				break;

			case ActivationType::ACTIVATION_RELU:
				this->activation = new ReLu(std::tuple<int, int, int>{1, 1, outputShape});
				this->heWeightsInit();
				break;

			case ActivationType::ACTIVATION_SOFTMAX:
				this->activation = new Softmax(outputShape);
				this->XGWeightsInit();
				break;

			default:
				perror("Dense Activation Type Error");
		}
	}

	~Output() {
		delete this->activation;
	}

	virtual void forward(xt::xarray<float> input) override;

	virtual void backward(
		float cost,
		float learningRate, 
        xt::xarray<int> trueLabel);

	void print() const override;

	void heWeightsInit();

	void XGWeightsInit();
};

#endif