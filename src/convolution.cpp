#include <iostream>
#include <vector>

#include "convolution.h"

// Algo in-place : inverse les lignes i et n-i-1 d'une matrice carrée
void reverseRows(xt::xarray<float> &matrix)
{
    int n = matrix.shape()[0];
    for (int i = 0; i < (n / 2); ++i)
    {
        for (int j = 0; j < n; ++j)
        {
            std::swap(matrix(i, j), matrix(n - i - 1, j));
        }
    }
}

// Algo in-place : inverse les éléments i et n-i-1 d'un vecteur de taille n
void reverseArray(xt::xarray<float> &arr, int k)
{
    int m = arr.shape()[0];
    int i = 0;
    int j = m - 1;

    while (i < j)
    {
        std::swap(arr(k, i), arr(k, j));
        ++i;
        --j;
    }
}

// Algo in-place : pivote une matrice carrée à 180 degrés
void rotateMatrix(xt::xarray<float> &kernel)
{
    int n = kernel.shape()[0];
    reverseRows(kernel);

    for (int i = 0; i < n; ++i)
    {
        reverseArray(kernel, i);
    }
}

// Renvoie le produit de convolution entre deux matrices de taille nxn
float prodConvolution(xt::xarray<float> input, xt::xarray<float> kernel)
{
    int n = input.shape()[0];
    float output = 0;
    rotateMatrix(kernel);

    auto d = input * kernel;

    for (int i = 0; i < n; ++i)
    {
        for (int j = 0; j < n; ++j)
        {
            output += d(i, j);
        }
    }

    return output;
}

// Algo in-place : pivote une matrice carrée à 180 degrés
xt::xarray<float> matrixConvolution(xt::xarray<float> &matrice, xt::xarray<float> &kernel)
{
    //
    int sizeKernel = kernel.shape()[0];

    int padding = 0;
    int stride = 1;

    int sizeNewMatrice = (matrice.shape()[0] - sizeKernel + 2 * padding) / stride + 1;

    xt::xarray<float> convolvedMatrice{xt::empty<uint8_t>({sizeNewMatrice, sizeNewMatrice})};

    int i = 0;
    int j = 0;

    while (i < sizeNewMatrice)
    {

        while (j < sizeNewMatrice)
        {    
            xt::xrange<int> rows(i, i + sizeKernel);
            xt::xrange<int> cols(j, j + sizeKernel);

            auto a = xt::view(matrice, rows, cols);
            convolvedMatrice(i, j) = prodConvolution(a, kernel);

            ++j;
        }
        j = 0;
        ++i;
    }

    return convolvedMatrice;
}
