#ifndef CONV_OP_H
#define CONV_OP_H

#include <xtensor/xarray.hpp>

// Algo in-place : pivote une matrice carrée à 180 degrés
void rotateMatrix(xt::xarray<float> &kernel);

void reverseRows(xt::xarray<float> &matrix);

void reverseArray(xt::xarray<float> &arr, int k);

// Renvoie le produit de convolution entre deux matrices de taille nxn
float prodConvolution(xt::xarray<float> input, xt::xarray<float> kernel);

// Algo in-place : fais la convolution d'une matrice par un kernel
xt::xarray<float> matrixConvolution(xt::xarray<float> matrice, xt::xarray<float> kernel, int stride, int padding);

// Effectue l'opération de crossCorrelation
xt::xarray<float> crossCorrelation(xt::xarray<float> matrice, xt::xarray<float> kernel, int stride, int padding);

// Aggrandis une matrice sur les côtés
xt::xarray<float> padMatrice(xt::xarray<float> matrice, int padding);

#endif