#include "network.h"

#include <chrono>
#include <fstream>
#include <iostream>
#include <istream>
#include <string>
#include <xtensor/xio.hpp>
#include <xtensor/xnpy.hpp>
#include <xtensor/xview.hpp>

#include "activation.h"
#include "const.h"
#include "convolution.h"
#include "dense.h"
#include "image.h"
#include "output.h"
#include "pooling.h"
#include "tools.h"

void NeuralNetwork::add(ILayer *layer) {
	this->nn.push_back(layer);
	return;
}

void NeuralNetwork::iter(xt::xarray<float> input, xt::xarray<int> label, float batchSize) {
	int nnSize = this->nn.size() - 1;
	this->nn[0]->forward(input);

	for (int i = 1; i < nnSize + 1; ++i) {
		this->nn[i]->forward(this->nn[i - 1]->output);
	}

	xt::xarray<float> recycling;

	recycling = lossFunction->prime(nn[nnSize]->output, label);

	recycling /= batchSize;

	for (int i = nnSize; i >= 0; --i) {
		if (this->nn[i]->name == "Output" || this->nn[i]->name == "Dense") {
			recycling = this->nn[i]->backward(recycling);
		} else {
			break;
		}
	}
}

void NeuralNetwork::batch(float size) {
	for (int i = this->nn.size() - 1; i >= 0; --i) {
		if (Output *output = dynamic_cast<Output *>(this->nn[i])) {
			output->weights = output->weights + (-learningRate) * (output->weightsGradient / size);
			output->bias = output->bias + (-learningRate) * (output->biasGradient / size);

			output->weightsGradient.fill(0.0);
			output->biasGradient.fill(0.0);

			if (output->normalize) {
				output->gammas = output->gammas + (-learningRate) * (output->gammasGradient / size);
				output->betas = output->betas + (-learningRate) * (output->betasGradient / size);

				output->gammasGradient.fill(0.0);
				output->betasGradient.fill(0.0);
			}

		} else if (Dense *dense = dynamic_cast<Dense *>(this->nn[i])) {
			dense->weights = dense->weights + (-learningRate) * (dense->weightsGradient / size);
			dense->bias = dense->bias + (-learningRate) * (dense->biasGradient / size);

			dense->weightsGradient.fill(0.0);
			dense->biasGradient.fill(0.0);

			if (dense->normalize) {
				dense->gammas = dense->gammas + (-learningRate) * (dense->gammasGradient / size);
				dense->betas = dense->betas + (-learningRate) * (dense->betasGradient / size);

				dense->gammasGradient.fill(0.0);
				dense->betasGradient.fill(0.0);
			}

		} else {
			break;
		}
	}
}

void NeuralNetwork::train(std::vector<std::tuple<xt::xarray<float>, xt::xarray<float>>> samples, std::vector<std::tuple<xt::xarray<float>, xt::xarray<float>>> validSamples, int epochs, int patience, float threshold) {
	if (batchSize > nbImagesTrain) {
		perror("BatchSize > totalNumberImage");
		exit(0);
	}

	// if (validationSplit != 0.0)	{
	// 	perror("Pas disponible !");
	// 	exit(0);
	// }

	// if (validationSplit != 0.0 && !validSamples.empty()) {
	// 	perror("Split validation != 0 avec des échantillons de validation");
	// 	exit(0);
	// }

	std::random_device rd;
	float trainLoss = 0.0;
	float minValidLoss = 1000000.0;
	float actualValidLoss = 0.0;
	int count1 = 0;
	int count2 = -1;
	int savedEpoch = 0;

	std::vector<float> validLossStored;
	std::vector<float> trainLossStored;

	float meanValidLoss = 0.0;
	float meanTrainLoss = 0.0;

	std::vector<std::tuple<xt::xarray<float>, xt::xarray<float>>> valid;
	std::vector<std::tuple<xt::xarray<float>, xt::xarray<float>>> train;

	int trainSize = 0;

	// if (validationSplit != 0.0) {
	// 	int ratio = nbImagesTrain * validationSplit;
	// 	int balance = ratio >> 1;

	// 	valid.reserve(ratio);
	// 	train.reserve(nbImagesTrain - ratio);
		
	// 	std::vector<std::tuple<xt::xarray<float>, xt::xarray<float>>> part1(samples.begin(), samples.begin() + balance);
	// 	std::vector<std::tuple<xt::xarray<float>, xt::xarray<float>>> part2(samples.end() - balance, samples.end());

	// 	std::vector<std::tuple<xt::xarray<float>, xt::xarray<float>>>::const_iterator first = samples.begin() + balance;
	// 	std::vector<std::tuple<xt::xarray<float>, xt::xarray<float>>>::const_iterator last = samples.end() - balance;

	// 	valid.insert(valid.end(), part1.begin(), part1.end());
	// 	valid.insert(valid.end(), part2.begin(), part2.end());

	// 	std::cout << valid.size() << std::endl;
	// 	std::cout << valid.capacity() << std::endl;

	// 	train.assign(first, last);

	// 	std::cout << train.size() << std::endl;
	// 	std::cout << train.capacity() << std::endl;
	// }

	train = std::move(samples);
	valid = std::move(validSamples);

	trainSize = train.size();

	xt::xarray<float> image = xt::empty<float>(IMAGE_TENSOR_DIM);
	xt::xarray<float> label;

	int r = trainSize % batchSize;

	std::cout << "Start training..." << std::endl;

	while (1) {
		if (this->shuffle) {
			std::mt19937 g(rd());
			std::shuffle(train.begin(), train.end(), g);
		}

		trainLoss = 0.0;

		std::cout << "nbEpoch: " << this->nbEpoch + 1 << std::endl;

		auto startTime = std::chrono::steady_clock::now();
		for (int i = 0; i < trainSize; i++) {
			image = std::get<0>(train[i]);
			label = std::get<1>(train[i]);

			this->iter(image, label, batchSize);

			trainLoss += lossFunction->compute(this->nn[this->nn.size() - 1]->output, label);

			if (i % batchSize == 0 && i != 0) {
				this->batch((float)batchSize);
				// std::cout << "loss actuelle: " << loss/(k+1.0) << std::endl;
			}
		}

		if (r > 0)	{
			this->batch((float)r);
		}

		auto endTime = std::chrono::steady_clock::now();
		auto duration = std::chrono::duration_cast<std::chrono::seconds>(endTime - startTime);

		nbEpoch++;
		trainLoss = trainLoss / (float)trainSize;

		std::cout << "loss: " << trainLoss << " (time: " << duration.count() << " sec)" << std::endl;

		std::cout << "Evaluation avec Validation" << std::endl;

		actualValidLoss = this->eval(valid);

		validLossStored.push_back(actualValidLoss);
		trainLossStored.push_back(trainLoss);

		meanTrainLoss = std::accumulate(trainLossStored.begin(), trainLossStored.end(), 0.0) / trainLossStored.size();
		meanValidLoss = std::accumulate(validLossStored.begin(), validLossStored.end(), 0.0) / validLossStored.size();

		if (actualValidLoss > minValidLoss) {
			count1++;

			if (count1 == patience) {
				std::cout << "Le seuil a été atteint, pas d'amélioration... STOP ?" << std::endl;
				std::cout << "Epoch: " << savedEpoch << " / Loss: " << this->loss << std::endl;

				if (confirm()) {
					break;
				}
				else	{
					count1 = 0;
				}
			}
		} else {
			minValidLoss = actualValidLoss;
			this->loss = trainLoss;
			this->save();
			savedEpoch = this->nbEpoch;
			count1 = 0;
		}

		if (std::fabs(actualValidLoss - trainLoss) > std::fabs(meanValidLoss - meanTrainLoss) && std::fabs(actualValidLoss - trainLoss) > threshold) {
			count2++;

			if (count2 == patience) {
				std::cout << "Le seuil a été atteint, overfitting?... STOP ?" << std::endl;

				if (confirm()) {
					break;
				}
				else	{
					count2 = 0;
				}
			}
		} else {
			count2 = 0;
		}

		if (nbEpoch % epochs == 0) {
			std::cout << "Le nombre d'époque a été atteint... STOP ?" << std::endl;

			if (confirm()) {
				break;
			}
		}

		std::cout << std::endl;
	}
}

float NeuralNetwork::eval(std::vector<std::tuple<xt::xarray<float>, xt::xarray<float>>> samples) {
	float eval = 0.0;
	float loss = 0.0;
	float accuracy = 0.0;
	int samplesSize = samples.size();
	xt::xarray<float> image;
	xt::xarray<float> label;

	int nnSize = this->nn.size();

	std::cout << "Start evaluation..." << std::endl;

	for (int i = 0; i < samplesSize; ++i) {
		image = std::get<0>(samples[i]);
		label = std::get<1>(samples[i]);

		this->nn[0]->forward(image, false);

		for (int j = 1; j < nnSize; ++j) {
			this->nn[j]->forward(this->nn[j - 1]->output, false);
		}

		if ((this->nn[this->nn.size() - 1]->output(0) < this->nn[this->nn.size() - 1]->output(1) && label(0) < label(1)) ||
			(this->nn[this->nn.size() - 1]->output(0) > this->nn[this->nn.size() - 1]->output(1) && label(0) > label(1))) {
			eval++;
		}

		loss += lossFunction->compute(this->nn[this->nn.size() - 1]->output, label);
	}

	loss = loss / samplesSize;
	accuracy = (eval / samplesSize) * 100.0;

	std::cout << "loss: " << loss << std::endl;
	std::cout << "accuracy: " << accuracy << '%' << std::endl;

	return loss;
}

// void NeuralNetwork::detect(xt::xarray<float> input) {

// }

void NeuralNetwork::load(const std::string path) {
	std::ifstream inputFile;
	std::ifstream layerFile;
	std::string tmpStr = path + "/nn" + ".dat";
	inputFile.open(tmpStr);

	int size = 0;
	std::string info;

	inputFile >> this->name;
	inputFile >> this->nbEpoch;
	inputFile >> size;
	inputFile >> this->batchSize;
	inputFile >> this->learningRate;
	inputFile >> info;
	inputFile >> this->loss;
	inputFile >> this->accuracy;
	inputFile >> this->validSplit;
	inputFile >> this->shuffle;

	if (!info.compare("MSE")) {
		this->lossFunction = new MSE();
	} else if (!info.compare("CrossEntropy")) {
		this->lossFunction = new CrossEntropy();
	} else {
		perror("Error with the type of loss function when loading");
		exit(0);
	}

	std::string buffer;

	for (int i = 0; i < size; ++i) {
		inputFile >> buffer;

		if (!buffer.find("conv")) {
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

			if (info.compare("ReLu")) {
				type = relu;
			} else if (info.compare("Softmax")) {
				type = softmax;
			} else if (info.compare("Sigmoid")) {
				type = sigmoid;
			} else if (info.compare("Activation")) {
				type = ACTIVATION_NO_TYPE;
			} else {
				perror("Error with the type of activation when loading (conv)");
				exit(0);
			}

			layerFile >> a;
			Convolution *conv = new Convolution{inputShape, filtersShape, type, (bool)a};

			tmpStr = path + "/" + buffer + "_filters.npy";

			conv->filters = xt::load_npy<float>(tmpStr);

			this->add(conv);

			layerFile.close();
		} else if (!buffer.find("pooling")) {
			tmpStr = path + "/" + buffer + ".dat";

			layerFile.open(tmpStr);
			int a, b, c, size, stride, padding;

			layerFile >> a >> b >> c;
			std::tuple<int, int, int> inputShape{a, b, c};
			layerFile >> size;
			layerFile >> stride;
			layerFile >> padding;
			layerFile >> info;

			PoolingType type;

			if (info.compare("max.")) {
				type = POOLING_MAX;
			} else if (info.compare("min.")) {
				type = POOLING_MIN;
			} else if (info.compare("avg.")) {
				type = POOLING_AVG;
			}
			else	{
				type = POOLING_NO_TYPE;
			}
			
			Pooling *pool = new Pooling{inputShape, size, stride, padding, type};

			this->add(pool);

			layerFile.close();
		} else if (!buffer.find("dense")) {
			tmpStr = path + "/" + buffer + ".dat";

			layerFile.open(tmpStr);

			int inputShape, outputShape, norm, flat, dropRate;
			layerFile >> inputShape;
			layerFile >> outputShape;

			ActivationType type;
			layerFile >> info;

			if (info.compare("ReLu")) {
				type = relu;
			} else if (info.compare("Softmax")) {
				type = softmax;
			} else if (info.compare("Sigmoid")) {
				type = sigmoid;
			} else if (info.compare("Activation")) {
				type = ACTIVATION_NO_TYPE;
			} else {
				perror("Error with the type of activation when loading (conv)");
				exit(0);
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
		} else if (!buffer.find("output")) {
			tmpStr = path + "/" + buffer + ".dat";

			layerFile.open(tmpStr);

			int inputShape, outputShape, norm, dropRate;
			layerFile >> inputShape;
			layerFile >> outputShape;

			ActivationType type;
			layerFile >> info;

			if (info.compare("ReLu")) {
				type = relu;
			} else if (info.compare("Softmax")) {
				type = softmax;
			} else if (info.compare("Sigmoid")) {
				type = sigmoid;
			} else if (info.compare("Activation")) {
				type = ACTIVATION_NO_TYPE;
			} else {
				perror("Error with the type of activation when loading (conv)");
				exit(0);
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

void NeuralNetwork::save() const {
	std::ofstream outputFile;

	std::ofstream nnFile;
	std::string path = savePath + '/' + this->name;

	try {
		std::filesystem::create_directories(path);
	} catch (const std::exception &e) {
		std::cerr << "Error creating directory: " << path << std::endl;
		return;
	}

	std::string tmpStr = path + "/nn" + ".dat";
	nnFile.open(tmpStr);

	nnFile << this->name << std::endl;
	nnFile << this->nbEpoch << std::endl;
	nnFile << this->nn.size() << std::endl;
	nnFile << this->batchSize << std::endl;
	nnFile << this->learningRate << std::endl;
	nnFile << this->lossFunction->name << std::endl;
	nnFile << this->loss << std::endl;
	nnFile << this->accuracy << std::endl;
	nnFile << this->validSplit << std::endl;
	nnFile << this->shuffle << std::endl;

	int nnSize = this->nn.size();

	for (int i = 0; i < nnSize; ++i) {
		if (Dense *dense = dynamic_cast<Dense *>(this->nn[i])) {
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
		} else if (Output *output = dynamic_cast<Output *>(this->nn[i])) {
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
			outputFile << pooling->size << std::endl;
			outputFile << pooling->stride << std::endl;
			outputFile << pooling->padding << std::endl;
			outputFile << pooling->type << std::endl;
			outputFile.close();
		} else {
			perror("Cette couche n'existe pas !");
			exit(0);
		}
	}

	nnFile.close();

	std::cout << "Saving done..." << std::endl;
}