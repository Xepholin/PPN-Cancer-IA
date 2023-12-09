#include <random>
#include <xtensor/xrandom.hpp>

#include "tools.h"

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

xt::xarray<float> flatten(xt::xarray<float> input)
{
    return xt::flatten(input);
}