/**
 * @file CrossEntropy.h
 * @brief Définition de la classe CrossEntropy.
 */

#ifndef CROSS_ENTROPY_H
#define CROSS_ENTROPY_H

#include "loss.h"

/**
 * @class CrossEntropy
 * @brief Classe représentant la fonction de perte Cross Entropy.
 */
class CrossEntropy : public Loss {
   public:
	int labelSize = 0;	// La taille des étiquettes.

	/**
	 * @brief Constructeur de la classe CrossEntropy.
	 *
	 * Initialise une instance de la fonction de perte Cross Entropy.
	 */
	CrossEntropy() {
		name = "CrossEntropy";
	}

	/**
	 * @brief Destructeur par défaut de la classe CrossEntropy.
	 */
	~CrossEntropy() = default;

	/**
	 * @brief Calcule la valeur de la fonction de perte Cross Entropy.
	 *
	 * @param output La sortie du réseau de neurones.
	 * @param label Les étiquettes réelles associées aux prédictions.
	 * @return La valeur de la fonction de perte Cross Entropy.
	 */
	virtual float compute(xt::xarray<float> output, xt::xarray<int> label) override;

	/**
	 * @brief Calcule le gradient de la fonction de perte Cross Entropy.
	 *
	 * @param output La sortie du réseau de neurones.
	 * @param label Les étiquettes réelles associées aux prédictions.
	 * @return Le gradient de la fonction de perte Cross Entropy.
	 */
	virtual xt::xarray<float> prime(xt::xarray<float> output, xt::xarray<int> label) override;
};

#endif	// CROSS_ENTROPY_H
