#include "network.h"

#include <fstream>
#include <iostream>
#include <istream>
#include <string>
#include <xtensor/xio.hpp>
#include <xtensor/xnpy.hpp>
#include <xtensor/xview.hpp>

#include "activation.h"
#include "convolution.h"
#include "dense.h"
#include "output.h"
#include "pooling.h"
#include "tools.h"

void NeuralNetwork::add(ILayer *layer) {
	this->nn.push_back(layer);
	return;
}

void NeuralNetwork::miniBatch(xt::xarray<float> batch, xt::xarray<int> trueLabels) {
	this->dropDense();

	for (int i = 1; i < batch.shape()[0]; ++i) {
		this->train(xt::view(batch, i), trueLabels);
	}
}

void NeuralNetwork::dropDense() {
	for (int i = 0; i < this->nn.size(); ++i) {
		if (Dense *dense = dynamic_cast<Dense *>(this->nn[i])) {
			dense->dropout(this->dropRate);
		}
	}
}

void NeuralNetwork::train(xt::xarray<float> input, xt::xarray<int> label) {
	this->nn[0]->forward(input);

	for (int i = 1; i < this->nn.size(); ++i) {
		this->nn[i]->forward(this->nn[i - 1]->output);
	}

	// std::cout << "output: " << this->nn[this->nn.size() - 1]->output << std::endl;

	float error = MSE(this->nn[this->nn.size() - 1]->output, label);

	// std::cout << error << std::endl;

	xt::xarray<float> recycling;

	for (int i = this->nn.size() - 1; i >= 0; --i) {
		if (this->nn[i]->name == "Output") {
			recycling = this->nn[i]->backward(label, this->learningRate);
		} else if (this->nn[i]->name == "Dense") {
			recycling = this->nn[i]->backward(recycling, this->learningRate);
		} else {
			break;
		}
	}
}

void NeuralNetwork::detect(xt::xarray<float> input) {}

void NeuralNetwork::load(const std::string path) {
	std::ifstream inputFile;
	std::ifstream layerFile;
	std::string tmpStr = path + "/nn" + ".dat";
	inputFile.open(tmpStr);

	int size = 0;
	std::string info;

	inputFile >> size;
	inputFile >> this->learningRate;
	inputFile >> this->dropRate;

	std::string buffer;

	for (int i = 0; i < size; ++i) {
		inputFile >> buffer;

		if (!buffer.find("conv")) {
			tmpStr = path + "/" + buffer + ".dat";

			layerFile.open(tmpStr);
			std::cout << tmpStr << std::endl;

			int a, b, c, d, e;
			layerFile >> a >> b >> c;
			std::tuple<int, int, int> inputShape{a, b, c};
			layerFile >> a >> b >> c;
			std::tuple<int, int, int> outputShape{a, b, c};
			layerFile >> a >> b >> c >> d >> e;
			std::tuple<int, int, int, int, int> filtersShape{a, b, c, d, e};

			ActivationType type;

			if (info.compare("ReLu")) {
				type = relu;
			} else {
				type = ACTIVATION_NO_TYPE;
			}

			layerFile >> a;
			Convolution *conv = new Convolution{1, inputShape, filtersShape, type, (bool)a};

			tmpStr = path + "/" + buffer + "_filters.npy";
			std::cout << tmpStr << std::endl;
			
			auto abcde = xt::load_npy<float>(tmpStr) ;
			conv->filters = abcde;
			this->add(conv);

			layerFile.close();
		}

		//  else if (!buffer.find("pooling")) {
		// 	tmpStr = path + "/" + buffer + ".dat";

		// 	layerFile.open(tmpStr);
		// 	std::cout << tmpStr << std::endl;

		// 	int a, b, c, d, e;
		// 	layerFile >> a >> b >> c;
		// 	std::tuple<int, int, int> inputShape{a, b, c};
		// 	layerFile >> a >> b >> c;
		// 	std::tuple<int, int, int> outputShape{a, b, c};
		// 	layerFile >> a >> b >> c >> d >> e;
		// 	std::tuple<int, int, int, int, int> filtersShape{a, b, c, d, e};

		// 	ActivationType type;

		// 	if (info.compare("ReLu")) {
		// 		type = relu;
		// 	} else {
		// 		type = ACTIVATION_NO_TYPE;
		// 	}

		// 	layerFile >> a;
		// 	Convolution *conv = new Convolution{1, inputShape, filtersShape, type, (bool)a};

		// 	tmpStr = path + "/" + buffer + "_filters.npy";
		// 	std::cout << tmpStr << std::endl;
		// 	auto abcde= xt::load_npy<float>(tmpStr);
		// 	layerFile.close();
		// }
	}

	inputFile.close();
}

void NeuralNetwork::save(const std::string path) const {
	std::ofstream outputFile;

	std::ofstream nnFile;
	std::string tmpStr = path + "/nn" + ".dat";
	nnFile.open(tmpStr);

	nnFile << this->nn.size() << std::endl;
	nnFile << this->learningRate << std::endl;
	nnFile << this->dropRate << std::endl;

	for (int i = 0; i < this->nn.size(); ++i) {
		if (Dense *dense = dynamic_cast<Dense *>(this->nn[i])) {
			tmpStr = path + "/dense" + std::to_string(i) + ".dat";
			nnFile << "dense" << std::to_string(i) << std::endl;
			outputFile.open(tmpStr);
			outputFile << dense->inputShape << std::endl;
			outputFile << dense->outputShape << std::endl;
			outputFile << dense->activation->name << std::endl;
			outputFile << dense->normalize << std::endl;
			outputFile << dense->flatten << std::endl;
			outputFile.close();

			tmpStr = path + "/dense" + std::to_string(i) + "_weights.npy";
			xt::dump_npy(tmpStr, dense->weights);

			tmpStr = path + "/dense" + std::to_string(i) + "_bias.npy";
			xt::dump_npy(tmpStr, dense->bias);
		} else if (Output *output = dynamic_cast<Output *>(this->nn[i])) {
			tmpStr = path + "/output" + std::to_string(i) + ".dat";
			nnFile << "output" << std::to_string(i) << std::endl;

			outputFile.open(tmpStr);

			outputFile << output->inputShape << std::endl;
			outputFile << output->outputShape << std::endl;
			outputFile << output->activation->name << std::endl;
			outputFile << output->normalize << std::endl;
			outputFile.close();

			tmpStr = path + "/output" + std::to_string(i) + "_weights.npy";
			xt::dump_npy(tmpStr, output->weights);

			tmpStr = path + "/output" + std::to_string(i) + "_bias.npy";
			xt::dump_npy(tmpStr, output->bias);

		} else if (Convolution *conv = dynamic_cast<Convolution *>(this->nn[i])) {
			tmpStr = path + "/conv" + std::to_string(i) + ".dat";
			outputFile.open(tmpStr);
			nnFile << "conv" << std::to_string(i) << std::endl;

			outputFile << std::get<0>(conv->inputShape) << ' ' << std::get<1>(conv->inputShape) << ' ' << std::get<2>(conv->inputShape) << std::endl;
			outputFile << std::get<0>(conv->outputShape) << ' ' << std::get<1>(conv->outputShape) << ' ' << std::get<2>(conv->outputShape) << std::endl;
			outputFile << std::get<0>(conv->filtersShape) << ' ' << std::get<1>(conv->filtersShape) << ' ' << std::get<2>(conv->filtersShape) << ' ' << std::get<3>(conv->filtersShape) << ' ' << std::get<4>(conv->filtersShape) << std::endl;

			outputFile << conv->activation->name << std::endl;
			outputFile << conv->normalize << std::endl;
			outputFile.close();

			tmpStr = path + "/conv" + std::to_string(i) + "_filters.npy";
			xt::dump_npy(tmpStr, conv->filters);
		}

		else if (Pooling *pooling = dynamic_cast<Pooling *>(this->nn[i])) {
			nnFile << "pooling" << std::to_string(i) << std::endl;

			tmpStr = path + "/pooling" + std::to_string(i) + ".dat";

			outputFile.open(tmpStr);

			outputFile << std::get<0>(pooling->inputShape) << ' ' << std::get<1>(pooling->inputShape) << ' ' << std::get<2>(pooling->inputShape) << std::endl;
			outputFile << std::get<0>(pooling->outputShape) << ' ' << std::get<1>(pooling->outputShape) << ' ' << std::get<2>(pooling->outputShape) << std::endl;
			outputFile << pooling->size << std::endl;
			outputFile << pooling->stride << std::endl;
			outputFile << pooling->padding << std::endl;
			outputFile.close();
		} else {
			perror("Cette couche n'existe pas !");
		}
	}

	nnFile.close();
}