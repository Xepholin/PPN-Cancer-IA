#ifndef MSE_H
#define MSE_H

#include "loss.h"

/**
 * @class MSE
 * @brief Classe représentant la fonction de perte Mean Squared Error (MSE).
 */
class MSE : public Loss {
   public:
	/**
	 * @brief Constructeur de la classe MSE.
	 *
	 * Initialise une instance de la fonction de perte MSE.
	 */
	MSE() {
		name = "MSE";
	}

	/**
	 * @brief Destructeur par défaut de la classe MSE.
	 */
	~MSE() = default;

	/**
	 * @brief Calcule la valeur de la fonction de perte Mean Squared Error (MSE).
	 *
	 * @param output La sortie du réseau de neurones.
	 * @param label Les étiquettes réelles associées aux prédictions.
	 * @return La valeur de la fonction de perte MSE.
	 */
	virtual float compute(xt::xarray<float> output, xt::xarray<int> label) override;

	/**
	 * @brief Calcule le gradient de la fonction de perte Mean Squared Error (MSE).
	 *
	 * @param output La sortie du réseau de neurones.
	 * @param label Les étiquettes réelles associées aux prédictions.
	 * @return Le gradient de la fonction de perte MSE.
	 */
	virtual xt::xarray<float> prime(xt::xarray<float> output, xt::xarray<int> label) override;
};

#endif	// MSE_H
