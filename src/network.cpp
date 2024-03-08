#include "network.h"

#include <fstream>
#include <iostream>
#include <istream>
#include <string>
#include <xtensor/xio.hpp>
#include <xtensor/xnpy.hpp>
#include <xtensor/xview.hpp>
#include <chrono>

#include "image.h"
#include "activation.h"
#include "convolution.h"
#include "dense.h"
#include "output.h"
#include "pooling.h"
#include "tools.h"
#include "const.h"

void NeuralNetwork::add(ILayer *layer)
{
	this->nn.push_back(layer);
	return;
}

void NeuralNetwork::iter(xt::xarray<float> input, xt::xarray<int> label) {
	int nnSize = this->nn.size() - 1;
	this->nn[0]->forward(input);

	for (int i = 1; i < nnSize + 1; ++i) {
		this->nn[i]->forward(this->nn[i - 1]->output);
	}

	xt::xarray<float> recycling;

	recycling = lossFunction->prime(nn[nnSize]->output, label);

	for (int i = nnSize; i >= 0; --i) {
		if (this->nn[i]->name == "Output" || this->nn[i]->name == "Dense") {
			recycling = this->nn[i]->backward(recycling, this->learningRate);
		}
		else	{
			break;
		}
	}
}

void NeuralNetwork::batch(int batchSize)	{
	for (int i = this->nn.size() - 1; i >= 0; --i)
	{
		if (Output *output = dynamic_cast<Output *>(this->nn[i]))
		{
			output->weights = output->weights + (-learningRate) * (output->weightsGradient / (float)batchSize);
			output->bias = output->bias + (-learningRate) * (output->biasGradient / (float)batchSize);

			output->weightsGradient.fill(0.0);
			output->biasGradient.fill(0.0);

			if (output->normalize)	{
				output->gammas = output->gammas + (-learningRate) * (output->gammasGradient / (float)batchSize);
				output->betas = output->betas + (-learningRate) * (output->betasGradient / (float)batchSize);

				output->gammasGradient.fill(0.0);
				output->betasGradient.fill(0.0);
			}
		}
		else if (Dense *dense = dynamic_cast<Dense *>(this->nn[i]))
		{
			dense->weights = dense->weights + (-learningRate) * (dense->weightsGradient / (float)batchSize);
			dense->bias = dense->bias + (-learningRate) * (dense->biasGradient / (float)batchSize);

			dense->weightsGradient.fill(0.0);
			dense->biasGradient.fill(0.0);

			if (dense->normalize)	{
				dense->gammas = dense->gammas + (-learningRate) * (dense->gammasGradient / (float)batchSize);
				dense->betas = dense->betas + (-learningRate) * (dense->betasGradient / (float)batchSize);

				dense->gammasGradient.fill(0.0);
				dense->betasGradient.fill(0.0);
			}
		}
		else	{
			break;
		}
	}
}

std::vector<std::tuple<int, float>> NeuralNetwork::train(const std::string path, int batchSize)
{
	if (batchSize > nbImagesTrain)	{
		perror("BatchSize > totalNumberImage");
	}

	std::vector<std::tuple<int, float>> result;

	std::string p0 = path + "/0";
	std::string p1 = path + "/1";

	xt::xarray<float> train0;
	xt::xarray<float> train1;

	int numImages = nbImagesTrain >> 1;

	std::cout << "Loading dataset..." << std::endl;

	if (PNGPBM == 0)	{
		train0 = importAllPNG(p0.c_str(), numImages);
		train1 = importAllPNG(p1.c_str(), numImages);
	}
	else	{
		train0 = importAllPBM(p0.c_str(), numImages);
		train1 = importAllPBM(p1.c_str(), numImages);
	}

	xt::xarray<float> image = xt::empty<float>(IMAGE_TENSOR_DIM);
	xt::xarray<float> label = xt::empty<float>({2});

	// std::string savePath = "../saves/" + this->name;

	// try
	// {
	// 	std::filesystem::create_directories(savePath);
	// }
	// catch (const std::exception &e)
	// {
	//  	std::cerr << "Error creating directory: " << savePath << std::endl;
	// }

	// this->save(savePath);

	std::cout << "Start training..." << std::endl;

	while (1)
	{
		xt::random::shuffle(train0);
		xt::random::shuffle(train1);

		std::cout << "nbEpoch: " << this->nbEpoch << std::endl;
		
		float loss = 0.0;

		auto startTime = std::chrono::steady_clock::now();

		for (int k = 0; k < nbImagesTrain; k++)
		{
			if (k & 1)
			{
				label = {0, 1};
				image = xt::view(train0, k >> 1);
			}
			else
			{
				label = {1, 0};
				image = xt::view(train1, k >> 1);
			}

			this->iter(image, label);

			loss += lossFunction->compute(this->nn[this->nn.size() - 1]->output, label);

			if (k % batchSize == 0 && k != 0)	{
				this->batch(batchSize);
				// std::cout << "loss actuelle: " << loss/(k+1.0) << std::endl;
			}
		}

		this->batch(batchSize);

		auto endTime = std::chrono::steady_clock::now();
		auto duration = std::chrono::duration_cast<std::chrono::seconds>(endTime - startTime);

		nbEpoch++;
		// this->save(savePath);
		
		std::cout << "loss: " << loss/(float)nbImagesTrain << "(time: " << duration.count() << " minutes)" << std::endl;

		if (PNGPBM == 0)	{
			this->eval("../assets/breast/eval");
		}
		else	{
			this->eval("../../processed1/eval");
		}

		// result.push_back(std::tuple<int, float>{nbEpoch, MSE(this->nn[this->nn.size() - 1]->output, label)});

		if (nbEpoch % 5 == 0 && !continueTraining())
		{
			break;
		}
	}

	return result;
}

void NeuralNetwork::eval(const std::string path)
{
	std::string p0 = path + "/0";
	std::string p1 = path + "/1";

	xt::xarray<float> eval0;
	xt::xarray<float> eval1;

	int numImage = nbImagesEval >> 1;

	std::cout << "Loading dataset..." << std::endl;

	if (PNGPBM == 0)	{
		eval0 = importAllPNG(p0.c_str(), numImage);
		eval1 = importAllPNG(p1.c_str(), numImage);
	}
	else	{
		eval0 = importAllPBM(p0.c_str(), numImage);
		eval1 = importAllPBM(p1.c_str(), numImage);
	}

	float eval = 0;

	xt::xarray<float> image = xt::empty<float>(IMAGE_TENSOR_DIM);

	std::cout << "Start evaluation..." << std::endl;

	std::cout << "Negative..." << std::endl;

	for (int i = 0; i < numImage; ++i)
	{
		image = xt::view(eval0, i);
		std::cout << eval0 << '\n' << std::endl;
		std::cout << image << '\n' << std::endl;
		exit(0);
		this->nn[0]->forward(image);

		for (int j = 1; j < this->nn.size(); ++j)
		{
			this->nn[j]->forward(this->nn[j - 1]->output);
		}

		// std::cout << this->nn[this->nn.size() - 1]->output<<std::endl;

		if (this->nn[this->nn.size() - 1]->output(0) < this->nn[this->nn.size() - 1]->output(1))
		{
			eval++;
		}
	}

	std::cout << "Positive..." << std::endl;

	for (int i = 0; i < numImage; ++i)
	{
		image = xt::view(eval1, i);
		this->nn[0]->forward(image);

		for (int j = 1; j < this->nn.size(); ++j)
		{
			this->nn[j]->forward(this->nn[j - 1]->output);
		}

		// std::cout << this->nn[this->nn.size() - 1]->output<<std::endl;
		
		if (this->nn[this->nn.size() - 1]->output(0) > this->nn[this->nn.size() - 1]->output(1))
		{
			eval++;
		}
	}

	float accuracy = (eval / nbImagesEval) * 100.0;
	std::cout << "accuracy : " << accuracy << "%\n" << std::endl;
	this->accuracy = accuracy;
}

void NeuralNetwork::detect(xt::xarray<float> input) {}

void NeuralNetwork::load(const std::string path)
{
	std::ifstream inputFile;
	std::ifstream layerFile;
	std::string tmpStr = path + "/nn" + ".dat";
	inputFile.open(tmpStr);

	int size = 0;
	std::string info;
	LossType lossType;

	inputFile >> this->name;
	inputFile >> this->nbEpoch;
	inputFile >> size;
	inputFile >> this->batchSize;
	inputFile >> this->learningRate;	
	inputFile >> info;
	inputFile >> this->accuracy;

	if (info.compare("MSE"))	{
		this->lossFunction = new MSE();
	}
	else if (info.compare("Cross Entropy"))	{
		this->lossFunction = new CrossEntropy();
	}
	else	{
		perror("Error with the type of loss function when loading");
	}

	std::string buffer;

	for (int i = 0; i < size; ++i)
	{
		inputFile >> buffer;

		if (!buffer.find("conv"))
		{
			tmpStr = path + "/" + buffer + ".dat";

			layerFile.open(tmpStr);

			int a, b, c, d, e;
			layerFile >> a >> b >> c;
			std::tuple<int, int, int> inputShape{a, b, c};
			layerFile >> a >> b >> c;
			std::tuple<int, int, int> outputShape{a, b, c};
			layerFile >> a >> b >> c >> d >> e;
			std::tuple<int, int, int, int, int> filtersShape{a, b, c, d, e};

			ActivationType type;
			layerFile >> info;

			if (info.compare("ReLu"))
			{
				type = relu;
			}
			else if (info.compare("Softmax"))
			{
				type = softmax;
			}
			else if (info.compare("Sigmoid"))
			{
				type = sigmoid;
			}
			else if (info.compare("Activation"))
			{
				type = ACTIVATION_NO_TYPE;
			}
			else	{
				perror("Error with the type of activation when loading (conv)");
			}

			layerFile >> a;
			Convolution *conv = new Convolution{1, inputShape, filtersShape, type, (bool)a};

			tmpStr = path + "/" + buffer + "_filters.npy";

			conv->filters = xt::load_npy<float>(tmpStr);

			this->add(conv);

			layerFile.close();
		}
		else if (!buffer.find("pooling"))
		{
			tmpStr = path + "/" + buffer + ".dat";

			layerFile.open(tmpStr);
			int a, b, c, size, stride, padding;

			layerFile >> a >> b >> c;
			std::tuple<int, int, int> inputShape{a, b, c};
			layerFile >> size;
			layerFile >> stride;
			layerFile >> padding;

			PoolingType type;

			if (info.compare("max."))
			{
				type = POOLING_MAX;
			}
			else if (info.compare("min."))
			{
				type = POOLING_MIN;
			}
			else if (info.compare("avg."))
			{
				type = POOLING_AVG;
			}
			layerFile >> a;
			Pooling *pool = new Pooling{inputShape, size, stride, padding, type};

			this->add(pool);

			layerFile.close();
		}
		else if (!buffer.find("dense"))
		{
			tmpStr = path + "/" + buffer + ".dat";

			layerFile.open(tmpStr);

			int inputShape, outputShape, norm, flat, dropRate;
			layerFile >> inputShape;
			layerFile >> outputShape;

			ActivationType type;
			layerFile >> info;

			if (info.compare("ReLu"))
			{
				type = relu;
			}
			else if (info.compare("Softmax"))
			{
				type = softmax;
			}
			else if (info.compare("Sigmoid"))
			{
				type = sigmoid;
			}
			else if (info.compare("Activation"))
			{
				type = ACTIVATION_NO_TYPE;
			}
			else	{
				perror("Error with the type of activation when loading (conv)");
			}

			layerFile >> dropRate;
			layerFile >> norm;
			layerFile >> flat;
			Dense *dense = new Dense{inputShape, outputShape, type, dropRate, (bool)norm, (bool)flat};

			tmpStr = path + "/" + buffer + "_weights.npy";
			dense->weights = xt::load_npy<float>(tmpStr);

			tmpStr = path + "/" + buffer + "_bias.npy";
			dense->bias = xt::load_npy<float>(tmpStr);
			this->add(dense);

			layerFile.close();
		}
		else if (!buffer.find("output"))
		{
			tmpStr = path + "/" + buffer + ".dat";

			layerFile.open(tmpStr);

			int inputShape, outputShape, norm, dropRate;
			layerFile >> inputShape;
			layerFile >> outputShape;

			ActivationType type;
			layerFile >> info;

			if (info.compare("ReLu"))
			{
				type = relu;
			}
			else if (info.compare("Softmax"))
			{
				type = softmax;
			}
			else if (info.compare("Sigmoid"))
			{
				type = sigmoid;
			}
			else if (info.compare("Activation"))
			{
				type = ACTIVATION_NO_TYPE;
			}
			else	{
				perror("Error with the type of activation when loading (conv)");
			}

			layerFile >> dropRate;
			layerFile >> norm;
			Output *out = new Output{inputShape, outputShape, type, dropRate, (bool)norm};

			tmpStr = path + "/" + buffer + "_weights.npy";

			out->weights = xt::load_npy<float>(tmpStr);

			tmpStr = path + "/" + buffer + "_bias.npy";

			out->bias = xt::load_npy<float>(tmpStr);

			this->add(out);

			layerFile.close();
		}
	}

	inputFile.close();

	std::cout << "Loading done..." << std::endl;
}

void NeuralNetwork::save(const std::string path) const
{
	std::ofstream outputFile;

	std::ofstream nnFile;
	std::string tmpStr = path + "/nn" + ".dat";
	nnFile.open(tmpStr);

	nnFile << this->name << std::endl;
	nnFile << this->nbEpoch << std::endl;
	nnFile << this->nn.size() << std::endl;
	nnFile << this->batchSize << std::endl;
	nnFile << this->learningRate << std::endl;
	nnFile << this->lossFunction->name << std::endl;
	nnFile << this->accuracy << std::endl;

	for (int i = 0; i < this->nn.size(); ++i)
	{
		if (Dense *dense = dynamic_cast<Dense *>(this->nn[i]))
		{
			tmpStr = path + "/dense" + std::to_string(i) + ".dat";
			nnFile << "dense" << std::to_string(i) << std::endl;
			outputFile.open(tmpStr);
			outputFile << dense->inputShape << std::endl;
			outputFile << dense->outputShape << std::endl;
			outputFile << dense->activation->name << std::endl;
			outputFile << dense->dropRate << std::endl;
			outputFile << dense->normalize << std::endl;
			outputFile << dense->flatten << std::endl;
			outputFile.close();

			tmpStr = path + "/dense" + std::to_string(i) + "_weights.npy";
			xt::dump_npy(tmpStr, dense->weights);

			tmpStr = path + "/dense" + std::to_string(i) + "_bias.npy";
			xt::dump_npy(tmpStr, dense->bias);
		}
		else if (Output *output = dynamic_cast<Output *>(this->nn[i]))
		{
			tmpStr = path + "/output" + std::to_string(i) + ".dat";
			nnFile << "output" << std::to_string(i) << std::endl;

			outputFile.open(tmpStr);

			outputFile << output->inputShape << std::endl;
			outputFile << output->outputShape << std::endl;
			outputFile << output->activation->name << std::endl;
			outputFile << output->dropRate << std::endl;
			outputFile << output->normalize << std::endl;
			outputFile.close();

			tmpStr = path + "/output" + std::to_string(i) + "_weights.npy";
			xt::dump_npy(tmpStr, output->weights);

			tmpStr = path + "/output" + std::to_string(i) + "_bias.npy";
			xt::dump_npy(tmpStr, output->bias);
		}
		else if (Convolution *conv = dynamic_cast<Convolution *>(this->nn[i]))
		{
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

		else if (Pooling *pooling = dynamic_cast<Pooling *>(this->nn[i]))
		{
			nnFile << "pooling" << std::to_string(i) << std::endl;

			tmpStr = path + "/pooling" + std::to_string(i) + ".dat";

			outputFile.open(tmpStr);

			outputFile << std::get<0>(pooling->inputShape) << ' ' << std::get<1>(pooling->inputShape) << ' ' << std::get<2>(pooling->inputShape) << std::endl;
			outputFile << pooling->size << std::endl;
			outputFile << pooling->stride << std::endl;
			outputFile << pooling->padding << std::endl;
			outputFile << pooling->type << std::endl;
			outputFile.close();
		}
		else
		{
			perror("Cette couche n'existe pas !");
		}
	}

	nnFile.close();

	std::cout << "Saving done..." << std::endl;
}