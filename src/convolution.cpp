#include <iostream>
#include <vector> 
#include <xtensor/xarray.hpp>
#include <xtensor/xio.hpp>
#include <xtensor/xrandom.hpp>


// Algo in-place : inverse les lignes i et n-i-1 d'une matrice carrée
void reverseRows(xt::xarray<int>& matrix)
{
    int n = matrix.shape()[0];
    for (int i = 0; i < (n/2); ++i)
    {
        for (int j = 0; j < n; ++j)
        {
            std::swap(matrix(i,j), matrix(n - i - 1,j));
        }
    }
}


// Algo in-place : inverse les éléments i et n-i-1 d'un vecteur de taille n
void reverseArray(xt::xarray<int>& arr, int k)
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
void rotateMatrix(xt::xarray<int>& kernel)
{
    int n = kernel.shape()[0];
    reverseRows(kernel);

    for (int i = 0; i<n; ++i)
    {
        reverseArray(kernel, i);
    }
    
}


// Renvoie le produit de convolution entre deux matrices de taille nxn
int convolMatrix(xt::xarray<int> input, xt::xarray<int> kernel)
{
    int n = input.shape()[0];
    int output = 0;
    rotateMatrix(kernel);
    auto d = input*kernel;

    for (int i = 0; i < n; ++i)
    {
        for (int j = 0; j < n; ++j)
        {
            output += d(i,j);
        }
    }
    


    return output;
}


int main() 
{
    xt::xarray<int> mat1 {{16,15,14,13},
                        {12,11,10,9},
                        {8,7,6,5},
                        {4,3,2,1}};

    xt::xarray<int> mat2 {{1,2,3,4},
                        {5,6,7,8},
                        {9,10,11,12},
                        {13,14,15,16}};

    int print = convolMatrix(mat1, mat2);
    std::cout << "Convolution value : " << print << std::endl;
    
    return 0;
}