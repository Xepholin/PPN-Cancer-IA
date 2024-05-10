#ifndef TOOLS_H
#define TOOLS_H

#include <filesystem>
#include <xtensor/xarray.hpp>

#include "network.h"

xt::xarray<float> kernelsGaussianDistro(int depth, int nbKernels, int height, int width);

/**
 * @brief Normalise les valeurs d'une matrice.
 *
 * @param input La matrice à normaliser.
 * @return La matrice normalisée.
 */
xt::xarray<float> normalized(xt::xarray<float> input);

/**
 * @brief Applatissement (flatten) d'une matrice en un vecteur.
 *
 * @param input La matrice à aplatir.
 * @return Le vecteur résultant de l'aplatissement.
 */
xt::xarray<float> flatten(xt::xarray<float> input);

int confirm();

/**
 * @brief Chargement d'un ensemble de données.
 *
 * @param path Le chemin du répertoire contenant les données.
 * @param label Les étiquettes associées aux données.
 * @param nbData Le nombre de données à charger.
 * @return Un vecteur de paires (entrée, étiquette).
 */
std::vector<std::tuple<xt::xarray<float>, xt::xarray<float>>> loadingSet(std::string path, xt::xarray<float> label, int nbData);

/**
 * @brief Chargement de plusieurs ensembles de données.
 *
 * @param path Le chemin du répertoire contenant les ensembles de données.
 * @param nbTotalData Le nombre total de données à charger.
 * @return Un vecteur de paires (entrée, étiquette).
 */
std::vector<std::tuple<xt::xarray<float>, xt::xarray<float>>> loadingSets(std::string path, int nbTotalData);

/**
 * @brief Affiche les informations sur un réseau de neurones.
 *
 * @param nn Le réseau de neurones à afficher.
 */
void display_network(NeuralNetwork nn);

xt::xarray<float> dot_product_fma(xt::xarray<float> weights, xt::xarray<float> input);

void setseed_xorshf96(void);

u_int32_t xorshf96(void);

#endif	// TOOLS_H
