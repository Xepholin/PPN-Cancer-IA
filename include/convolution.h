#ifndef CONVOLUTION_H
#define CONVOLUTION_H

#include <xtensor/xarray.hpp>
#include <xtensor/xio.hpp>
#include <xtensor/xrandom.hpp>

// Algo in-place : inverse les lignes i et n-i-1 d'une matrice carrée
void reverseRows(xt::xarray<float> &matrix);

// Algo in-place : inverse les éléments i et n-i-1 d'un vecteur de taille n
void reverseArray(xt::xarray<float> &arr, int k);

// Algo in-place : pivote une matrice carrée à 180 degrés
void rotateMatrix(xt::xarray<float> &kernel);

// Renvoie le produit de convolution entre deux matrices de taille nxn
float prodConvolution(xt::xarray<float> input, xt::xarray<float> kernel);

// Algo in-place : fais la convolution d'une matrice par un kernel
xt::xarray<float> matrixConvolution(xt::xarray<float> &matrice, xt::xarray<float> &kernel);

#endif