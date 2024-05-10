#ifndef CONV_OP_H
#define CONV_OP_H

#include <xtensor/xarray.hpp>

/**
 * @brief Pivote une matrice carrée à 180 degrés.
 *
 * @param kernel La matrice carrée à pivoter.
 */
void rotateMatrix(xt::xarray<float> &kernel);

/**
 * @brief Inverse les lignes d'une matrice.
 *
 * @param matrix La matrice dont les lignes doivent être inversées.
 */
void reverseRows(xt::xarray<float> &matrix);

/**
 * @brief Inverse les éléments d'un tableau sur les 'k' premières positions.
 *
 * @param arr Le tableau à inverser.
 * @param k Le nombre d'éléments à inverser.
 */
void reverseArray(xt::xarray<float> &arr, int k);

/**
 * @brief Calcule le produit de convolution entre deux matrices de taille nxn.
 *
 * @param input La matrice d'entrée.
 * @param kernel Le filtre de convolution.
 * @return Le résultat de la convolution.
 */
float prodConvolution(xt::xarray<float> input, xt::xarray<float> kernel);

/**
 * @brief Applique l'opération de convolution entre une matrice et un filtre.
 *
 * @param matrice La matrice sur laquelle appliquer la convolution.
 * @param kernel Le filtre de convolution.
 * @param stride Le décalage entre les filtres lors de l'application de la convolution.
 * @param padding Le rembourrage à appliquer autour de la matrice avant la convolution.
 * @return Le résultat de la convolution.
 */
xt::xarray<float> matrixConvolution(xt::xarray<float> matrice, xt::xarray<float> kernel, int stride, int padding);

/**
 * @brief Applique l'opération de cross-correlation entre une matrice et un filtre.
 *
 * @param matrice La matrice sur laquelle appliquer la cross-correlation.
 * @param kernel Le filtre de cross-correlation.
 * @param stride Le décalage entre les filtres lors de l'application de la cross-correlation.
 * @param padding Le rembourrage à appliquer autour de la matrice avant la cross-correlation.
 * @return Le résultat de la cross-correlation.
 */
xt::xarray<float> crossCorrelation(xt::xarray<float> matrice, xt::xarray<float> kernel, int stride, int padding);

/**
 * @brief Aggrandit une matrice en ajoutant du rembourrage autour de ses côtés.
 *
 * @param matrice La matrice à agrandir.
 * @param padding Le nombre de pixels à ajouter autour de la matrice.
 * @return La matrice agrandie.
 */
xt::xarray<float> padMatrice(xt::xarray<float> matrice, int padding);

#endif	// CONV_OP_H
