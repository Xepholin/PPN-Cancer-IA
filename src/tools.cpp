#include <random>
#include <xtensor/xrandom.hpp>
#include <cmath>
#include <filesystem>
#include <immintrin.h>

#include "tools.h"
#include "layer.h"

#include "convolution.h"
#include "dense.h"
#include "pooling.h"
#include "output.h"
#include "image.h"
#include "const.h"

xt::xarray<float> kernelsGaussianDistro(int depth, int nbKernels, int height, int width)
{
	// Pour générer une simulation d'une loi centrée et réduite
	std::default_random_engine generator;
	std::normal_distribution<float> ech(0.0, 1.0);
	float random = 0.0;

	// Crée une matrice echantillon de loi normale centrée et réduite
	xt::xarray<float> kernels = xt::empty<float>({depth, nbKernels, height, width});

	for (int i = 0; i < depth; ++i)
	{
		for (int j = 0; j < nbKernels; ++j)
		{
			for (int k = 0; k < height; ++k)
			{
				for (int m = 0; m < width; ++m)
				{
					while (random < 0.0 || random > 1.0)
					{
						random = ech(generator);
					}

					kernels(i, j, k, m) = random;
				}
			}
		}
	}

	return kernels;
}

xt::xarray<float> dot_product_fma(xt::xarray<float> weights, xt::xarray<float> input)
{

	int inputShape = weights.shape()[1];
	int outputShape = weights.shape()[0];

	float *result_arr = (float *)_mm_malloc(8 * sizeof(float), 32); // Align to 32 bytes
	xt::xarray<float> result = xt::zeros<float>({outputShape});

	for (int j = 0; j < outputShape; ++j)
	{
		int k = 0;
		__m256 sum = _mm256_setzero_ps(); // Initialize sum to zero
		for (int i = 8; i < inputShape; i += 8)
		{						
			__m256 weight_vect = _mm256_loadu_ps(&weights(j, i) - 8); // Load 8 elements from array b
			__m256 input_vect = _mm256_loadu_ps(&input(i) - 8);		  // Load 8 elements from array b

			sum = _mm256_fmadd_ps(weight_vect, input_vect, sum); // Fused multiply-add operation
			++k;
		}

		if (k * 8 != inputShape)
		{
			float dotResult = 0;
			for (int i = k*8; i < inputShape; ++i)
			{
				dotResult += weights(j, i) * input(i);
			}
			result(j)+=dotResult;

		}

		// Horizontal addition of the eight 32-bit floats into a single __m256 value
		__m256 shuf = _mm256_hadd_ps(sum, sum);
		shuf = _mm256_hadd_ps(shuf, shuf);
		// Store the __m256 vector into a float array and extract the result

		_mm256_store_ps(result_arr, shuf);
		// Extract the result from the first element of the array
		result(j) += result_arr[0] + result_arr[4];
	}

	free(result_arr);
	return result;
}

xt::xarray<float> normalized(xt::xarray<float> input)
{

	float mean = xt::mean(input)();
	float variance = xt::variance(input)();

	xt::xarray<float> normalized = (input - mean) / std::sqrt(variance + 1e-6);

	return normalized;
}

int confirm()
{
	std::string path;
	std::string save;

	while (1)
	{
		std::cout << "confirmation ? [y/n]" << std::endl;
		getline(std::cin, save);
		std::transform(save.begin(), save.end(), save.begin(),
					   [](unsigned char c)
					   { return std::tolower(c); });

		if (!save.compare("y"))
		{
			return 1;
		}
		else if (!save.compare("n"))
		{
			return 0;
		}
		else
		{
			continue;
		}
	}
}

std::vector<std::tuple<xt::xarray<float>, xt::xarray<float>>> loadingSet(std::string path, xt::xarray<float> label, int nbData)	{
	xt::xarray<float> dataset;
	xt::xarray<float> image;
	std::vector<std::tuple<xt::xarray<float>, xt::xarray<float>>> set;

	if (PNGPBM == 0)	{
		dataset = importAllPNG(path.c_str(), nbData);
	}
	else	{
		dataset = importAllPBM(path.c_str(), nbData);
	}

	xt::random::shuffle(dataset);

	std::tuple<xt::xarray<float>, xt::xarray<float>> stored;

	int datasetSize = dataset.shape()[0];

	for(int i = 0; i < datasetSize; ++i)	{
		image = xt::view(dataset, i);

		stored = {image, label};
		set.push_back(stored);
	}

	return set;
}

std::vector<std::tuple<xt::xarray<float>, xt::xarray<float>>> loadingSets(std::string path, int nbTotalData)	{
	std::vector<std::tuple<xt::xarray<float>, xt::xarray<float>>> sets;

	std::vector<std::tuple<xt::xarray<float>, xt::xarray<float>>> set0;
	std::vector<std::tuple<xt::xarray<float>, xt::xarray<float>>> set1;

	int nbData = nbTotalData >> 1;

	std::cout << "Loading dataset..." << std::endl;

	set0 = loadingSet(path + "/0", {0, 1}, nbData);
	set1 = loadingSet(path + "/1", {1, 0}, nbData);

	sets.reserve(set0.size() + set1.size());
	sets.insert(sets.end(), set0.begin(), set0.end());
	sets.insert(sets.end(), set1.begin(), set1.end());

	return sets;
}

void display_network(NeuralNetwork nn)
{
	int nnSize = nn.nn.size();
	for (int i = 0; i < nnSize; ++i)
	{
		std::cout << nn.nn[i]->name << std::endl;
		if (Convolution *conv = dynamic_cast<Convolution *>(nn.nn[i]))
		{
			std::cout << "   " << std::get<0>(conv->inputShape) << " " << std::get<1>(conv->inputShape) << " " << std::get<2>(conv->inputShape) << " " << std::endl;
			std::cout << "   " << std::get<0>(conv->outputShape) << " " << std::get<1>(conv->outputShape) << " " << std::get<2>(conv->outputShape) << " " << std::endl;
			std::cout << "   " << std::get<0>(conv->filtersShape) << " " << std::get<1>(conv->filtersShape) << " " << std::get<2>(conv->filtersShape) << " " << std::get<3>(conv->filtersShape) << " " << std::get<4>(conv->filtersShape) << std::endl;
			std::cout << "   " << conv->activation->name << std::endl;
			std::cout << "   " << conv->normalize << std::endl;
		}
		else if (Pooling *pool = dynamic_cast<Pooling *>(nn.nn[i]))
		{
			std::cout << "   " << pool->size << std::endl;
			std::cout << "   " << pool->stride << std::endl;
			std::cout << "   " << pool->padding << std::endl;
		}
		else if (Output *outp = dynamic_cast<Output *>(nn.nn[i]))
		{
			std::cout << "   " << outp->inputShape << std::endl;
			std::cout << "   " << outp->outputShape << std::endl;
			std::cout << "   " << outp->activation->name << std::endl;
		}
		else if (Dense *dense = dynamic_cast<Dense *>(nn.nn[i]))
		{
			std::cout << "   " << dense->inputShape << std::endl;
			std::cout << "   " << dense->outputShape << std::endl;
			std::cout << "   " << dense->activation->name << std::endl;
		}
	}
}


static unsigned long xx = 123456789, yy = 362436069, zz = 521288629;

void setseed_xorshf96(void)
{
	srand(time(NULL));
	xx = rand();
	yy = rand();
	zz = rand();
}

u_int32_t xorshf96(void)
{ // period 2^96-1
	uint32_t t;
	xx ^= xx << 16;
	xx ^= xx >> 5;
	xx ^= xx << 1;

	t = xx;
	xx = yy;
	yy = zz;
	zz = t ^ xx ^ yy;

	return (u_int32_t) zz;
}
