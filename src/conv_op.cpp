#include <xtensor/xview.hpp>
#include <xtensor/xrandom.hpp>
#include <xtensor/xio.hpp>

#include "conv_op.h"

// Algo in-place : inverse les lignes i et n-i-1 d'une matrice carrée
void reverseRows(xt::xarray<float> &matrix)
{
    int n = matrix.shape()[0];

    for (int i = 0; i < (n >> 1); ++i)
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

// Renvoie le produit de la correlation croisée entre entre deux matrices de taille nxn
float prodCrossCorelation(xt::xarray<float> input, xt::xarray<float> kernel)
{
    int n = input.shape()[0];
    float output = 0;

    xt::xarray<float> d = input * kernel;

    for (int i = 0; i < n; ++i)
    {
        for (int j = 0; j < n; ++j)
        {
            output += d(i, j);
        }
    }

    return output;
}

// Effectue l'opération de crossCorrelation
xt::xarray<float> crossCorrelation(xt::xarray<float> matrice, xt::xarray<float> kernel, int stride, int padding)
{

    int sizeKernelX = kernel.shape()[0];
    int sizeKernelY = kernel.shape()[1];

    int sizeNewMatriceX = (matrice.shape()[0] - sizeKernelX + 2 * padding) / stride + 1;
    int sizeNewMatriceY = (matrice.shape()[1] - sizeKernelY + 2 * padding) / stride + 1;

    xt::xarray<float> crossCorrelationMatrice{xt::empty<uint8_t>({sizeNewMatriceX, sizeNewMatriceY})};
	xt::xarray<float> a;

    for(int i = 0 ; i < sizeNewMatriceX; ++i)
    {
		xt::xrange<int> rows(i, i + sizeKernelX);
		
        for(int j = 0 ; j < sizeNewMatriceY; ++j)
        {    
            xt::xrange<int> cols(j, j + sizeKernelY);

            a = xt::view(matrice, rows, cols);
            crossCorrelationMatrice(i, j) = prodCrossCorelation(a, kernel);

        }
    }

    return crossCorrelationMatrice;
}

xt::xarray<float> matrixConvolution(xt::xarray<float> matrice, xt::xarray<float> kernel, int stride, int padding)
{
    rotateMatrix(kernel);
    return crossCorrelation(matrice, kernel, stride, padding );
}


xt::xarray<float> padMatrice(xt::xarray<float>  matrice, int padding){

	int shape0 = matrice.shape()[0];
	int shape1 = matrice.shape()[1];

    int sizeNewMatriceX = shape0 + 2 * padding;
    int sizeNewMatriceY = shape1 + 2 * padding;

    xt::xarray<float> paddedMatrice{xt::zeros<uint8_t>({sizeNewMatriceX, sizeNewMatriceY})};

    for (int i = 0; i < shape0; ++i)
    {
        for (int j = 0; j < shape1; ++j)
        {
            paddedMatrice(i+padding, j+padding) = matrice(i, j);
        }
    }

    return paddedMatrice;
}
