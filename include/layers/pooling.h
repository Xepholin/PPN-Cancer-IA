#ifndef POOLING_H
#define POOLING_H

#include <tuple>
#include <xtensor/xarray.hpp>

#include "layer.h"

/**
 * @enum PoolingType
 * @brief Enumération des différents types de pooling.
 */
enum PoolingType {
	POOLING_NO_TYPE,  // Pas de pooling.
	POOLING_MAX,	  // Pooling maximum.
	POOLING_MIN,	  // Pooling minimum.
	POOLING_AVG		  // Pooling moyen.
};

/**
 * @brief Opérateur de sortie pour PoolingType.
 *
 * @param out Le flux de sortie.
 * @param value Le type de pooling à afficher.
 * @return Le flux de sortie mis à jour.
 */
std::ostream& operator<<(std::ostream& out, const PoolingType value);

/**
 * @class Pooling
 * @brief Classe représentant une couche de pooling.
 */
class Pooling : public ILayer {
   public:
	std::tuple<int, int, int> inputShape{0, 0, 0};	 // La forme de l'entrée.
	std::tuple<int, int, int> outputShape{0, 0, 0};	 // La forme de la sortie.

	int depth = 0;									  // La profondeur de l'entrée.
	int size = 1;									  // La taille du filtre de pooling.
	int stride = 1;									  // Le décalage entre les filtres lors de l'application du pooling.
	int padding = 0;								  // Le rembourrage à appliquer autour de l'entrée avant le pooling.
	PoolingType type = PoolingType::POOLING_NO_TYPE;  // Le type de pooling à appliquer.

	/**
	 * @brief Constructeur de la classe Pooling.
	 *
	 * Initialise une couche de pooling avec les paramètres spécifiés.
	 *
	 * @param inputShape La forme de l'entrée avec les dimensions (profondeur, hauteur, largeur).
	 * @param size La taille du filtre de pooling.
	 * @param stride Le décalage entre les filtres lors de l'application du pooling.
	 * @param padding Le padding à appliquer autour de l'entrée avant le pooling.
	 * @param poolingType Le type de pooling à appliquer (par défaut, pas de pooling).
	 */
	Pooling(std::tuple<int, int, int> inputShape,
			int size, int stride, int padding,
			PoolingType poolingType = PoolingType::POOLING_NO_TYPE) {
		name = "Pooling";

		this->inputShape = inputShape;

		this->depth = std::get<0>(inputShape);
		int height = std::get<1>(inputShape);
		int width = std::get<2>(inputShape);

		int outputHeight = (height - size + 2 * padding) / stride + 1;
		int outputWidth = (width - size + 2 * padding) / stride + 1;

		this->outputShape = std::tuple<int, int, int>(depth, outputHeight, outputWidth);

		this->input = xt::empty<float>({depth, height, width});
		this->output = xt::empty<float>({depth, outputHeight, outputWidth});

		this->size = size;
		this->stride = stride;
		this->padding = padding;
		this->type = poolingType;
	}

	/**
	 * @brief Destructeur par défaut de la classe Pooling.
	 */
	~Pooling() = default;

	/**
	 * @brief Fonction de propagation avant de la couche de pooling.
	 *
	 * Calcule la sortie de la couche de pooling pour une entrée donnée.
	 *
	 * @param input L'entrée de la couche de pooling.
	 * @param training Indique si le réseau est en phase d'entraînement (par défaut, true).
	 */
	void forward(xt::xarray<float> input, bool training = true) override;

	/**
	 * @brief Fonction de propagation arrière de la couche de pooling.
	 *
	 * Calcule le gradient de l'erreur par rapport à l'entrée de la couche de pooling.
	 *
	 * @param gradient Le gradient de l'erreur par rapport à la sortie de la couche suivante.
	 * @return Le gradient de l'erreur par rapport à l'entrée de la couche de pooling.
	 */
	xt::xarray<float> backward(xt::xarray<float> gradient) override;

	void print() const override;

	/**
	 * @brief Applique l'opération de pooling sur une matrice.
	 *
	 * @param matrix La matrice sur laquelle appliquer l'opération de pooling.
	 * @return La matrice après l'opération de pooling.
	 */
	float pooling(xt::xarray<float> matrix);

	xt::xarray<float> poolingMatrice(xt::xarray<float> matrix);
};

#endif	// POOLING_H
