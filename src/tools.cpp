#include <random>
#include <xtensor/xrandom.hpp>
#include <cmath>
#include <filesystem>

#include "tools.h"
#include "layer.h"

xt::xarray<float> kernelsGaussianDistro(int depth, int nbKernels, int height, int width)
{
    // Pour générer une simulation d'une loi centrée et réduite 
    std::default_random_engine generator;
    std::normal_distribution<float> ech(0.0, 1.0);

    // Crée une matrice echantillon de loi normale centrée et réduite
    xt::xarray<float> kernels = xt::empty<float>({depth, nbKernels, height, width});
    
    for (int i = 0; i < depth; ++i)
    {
        for (int j = 0; j < nbKernels; ++j)
        {
            for (int k = 0; k < height; ++k)
            {
                for (int m = 0; m < width; ++m) {

                    float random = ech(generator);

                    while(random < 0.0 || random > 1.0)    {
                        random = ech(generator);
                    }
                    
                    kernels(i, j, k, m) = random;
                }
            }

        }
    }
    
    return kernels;

}

xt::xarray<float> normalized(xt::xarray<float> input)
{

    float mean = xt::mean(input)();
    float variance = xt::variance(input)();

    xt::xarray<float> normalized = (input - mean) / std::sqrt(variance + 1e-6);

    return normalized;
}

xt::xarray<float> flatten(xt::xarray<float> input)
{
    return xt::flatten(input);
}

float MSE(xt::xarray<float> output, xt::xarray<int> trueValue)
{
    float err = 0.0;
    for (int i = 0; i < output.size(); ++i)
    {
        err += 0.5 * ((output(i) - trueValue(i)) * (output(i) - trueValue(i)));
    }

    return err;
}

float crossEntropy(xt::xarray<float> output, xt::xarray<int> trueValue)
{
    float err = 0.0;
    xt::xarray<float> outputLog = xt::log(output);
    for (int i = 0; i < output.size(); ++i)
    {
        err -= trueValue(i) * outputLog(i);
    }

    return err;
}

int continueTraining()	{
	int train = 0;

	while(1)	{
		std::string stop;

		std::cout << "continue ? [y/n]" << std::endl;
		getline (std::cin, stop);
		std::transform(stop.begin(), stop.end(), stop.begin(),
		[](unsigned char c){ return std::tolower(c); });

		if (!stop.compare("y"))	{
			train = 1;
			break;
		}
		else if (!stop.compare("n"))	{
			train = 0;
			break;
		}
		else	{
			std::cout << "mauvaise commande" << std::endl;
			continue;
		}
	}

	return train;
}

void saveConfirm(NeuralNetwork nn)	{
	while(1)	{
		std::string save;

		std::cout << "save ? [y/n]" << std::endl;
		getline (std::cin, save);
		std::transform(save.begin(), save.end(), save.begin(),
		[](unsigned char c){ return std::tolower(c); });

		if (!save.compare("y"))	{
			std::string path;

			std::cout << "enter name" << std::endl;
			getline(std::cin, path);

			path = "../saves/" + path;
			
			try
			{
				std::filesystem::create_directories(path);
			}
			catch (const std::exception &e)
			{
				std::cerr << "Error creating directory: " << path << std::endl;
				return;
			}

			nn.save(path);
			break;
		}
		else if (!save.compare("n"))	{
			break;
		}
		else	{
			continue;
		}
	}
}
