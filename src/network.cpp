#include "network.h"

#include <istream>
#include <fstream>
#include <iostream>
#include <xtensor/xio.hpp>
#include <xtensor/xview.hpp>
#include <xtensor/xnpy.hpp>
#include <string>

#include "pooling.h"
#include "convolution.h"
#include "dense.h"
#include "output.h"
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
		}
		else if (this->nn[i]->name == "Dense") {
			recycling = this->nn[i]->backward(recycling, this->learningRate);
		}
		else	{
			break;
		}
	}
}

void NeuralNetwork::detect(xt::xarray<float> input) {}

void NeuralNetwork::load(const std::string path) {

	std::ifstream inputFile;
	inputFile.open(path);

	for (int i = 0; i < this->nn.size(); ++i)	{
		
	}
}

void NeuralNetwork::save(const std::string path) const {

	std::ofstream outputFile;
	std::string tmpStr =  path + "/nn" +".dat";
	outputFile.open(tmpStr);
	outputFile << this->learningRate << std::endl;
	outputFile << this->dropRate << std::endl;
	outputFile.close();

	for (int i = 0; i < this->nn.size(); ++i)	{
		
		if (Dense *dense = dynamic_cast<Dense *>(this->nn[i]))	{
			tmpStr =  path + "/dense" + std::to_string(i) +".dat";
			
			outputFile.open(tmpStr);
			outputFile << dense->inputShape << std::endl;
			outputFile << dense->outputShape << std::endl;
			outputFile << dense->activation->name << std::endl;
			outputFile << dense->normalize << std::endl;
			outputFile << dense->flatten << std::endl;
			outputFile.close();
			
			tmpStr =  path + "/dense" + std::to_string(i) + "_weights.npy";
			xt::dump_npy(tmpStr, dense->weights);	

			tmpStr =  path + "/dense" + std::to_string(i) + "_bias.npy";
			xt::dump_npy(tmpStr, dense->bias);	
		}
		else if (Output *output= dynamic_cast<Output *>(this->nn[i]))	{
			tmpStr =  path + "/output" + std::to_string(i) + ".dat";
			outputFile.open(tmpStr);

			outputFile << output->inputShape << std::endl;
			outputFile << output->outputShape << std::endl ;
			outputFile << output->activation->name << std::endl;
			outputFile << output->normalize << std::endl;
			outputFile.close();
			
			tmpStr =  path + "/output" + std::to_string(i) + "_weights.npy";
			xt::dump_npy(tmpStr, output->weights);

			tmpStr =  path + "/output" + std::to_string(i) + "_bias.npy";
			xt::dump_npy(tmpStr, output->bias);

		}
		else if (Convolution *conv = dynamic_cast<Convolution *>(this->nn[i]))	{
			
			tmpStr =  path + "/conv" + std::to_string(i) + ".dat";
			outputFile.open(tmpStr);

			outputFile << std::get<0>(conv->inputShape) << ' ' << std::get<1>(conv->inputShape) << ' ' << std::get<2>(conv->inputShape)<< std::endl;
			outputFile << std::get<0>(conv->outputShape) << ' ' << std::get<1>(conv->outputShape) << ' ' << std::get<2>(conv->outputShape)<< std::endl;
			outputFile << std::get<0>(conv->filtersShape) << ' ' << std::get<1>(conv->filtersShape) << ' '  << std::get<2>(conv->filtersShape) << ' ' << std::get<3>(conv->filtersShape) << ' ' << std::get<4>(conv->filtersShape) << std::endl;
			
			outputFile << conv->activation->name << std::endl;
			outputFile << conv->normalize << std::endl;
			outputFile.close();
			
			tmpStr =  path + "/conv" + std::to_string(i) + "_filters.npy";
			xt::dump_npy(tmpStr, conv->filters);
		}

		else if (Pooling *pooling = dynamic_cast<Pooling *>(this->nn[i]))	{
			
			tmpStr =  path + "/pooling" + std::to_string(i) + ".dat";
			outputFile.open(tmpStr);

			outputFile << std::get<0>(pooling->inputShape) << ' ' << std::get<1>(pooling->inputShape) << ' ' << std::get<2>(pooling->inputShape)<< std::endl;
			outputFile << std::get<0>(pooling->outputShape) << ' ' << std::get<1>(pooling->outputShape) << ' ' << std::get<2>(pooling->outputShape)<< std::endl;
			outputFile << pooling->size << std::endl;
			outputFile << pooling->stride << std::endl;
			outputFile << pooling->padding << std::endl;
			outputFile.close();
		}
		else	{
			perror("Cette couche n'existe pas !");
		}
	}
}