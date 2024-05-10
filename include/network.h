/**
 * @file network.h
 * @brief Définition de la classe NeuralNetwork pour un réseau de neurones.
 */

#ifndef NETWORK_H
#define NETWORK_H

#include <vector>
#include <xtensor/xarray.hpp>
#include <xtensor/xrandom.hpp>

#include "crossEntropy.h"
#include "layer.h"
#include "loss.h"
#include "mse.h"

/**
 * @brief Classe représentant un réseau de neurones.
 */
class NeuralNetwork {
   public:
	std::string name;		   // Nom du réseau de neurones.
	std::vector<ILayer *> nn;  // Couches du réseau de neurones.

	float learningRate;		 // Taux d'apprentissage du réseau.
	int nbEpoch = 0;		 // Nombre d'époques d'entraînement.
	float loss = 1000000.0;	 // Valeur de la fonction de perte.
	float accuracy = 0.0;	 // Précision du réseau.
	int batchSize = 1;		 // Taille des lots (batch) lors de l'entraînement.
	float validSplit = 0.2;	 // Proportion des données à utiliser pour la validation.
	bool shuffle = true;	 // Indique si les données doivent être mélangées avant l'entraînement.

	Loss *lossFunction;	 // Fonction de perte utilisée par le réseau.

	/**
	 * @brief Constructeur par défaut de la classe NeuralNetwork.
	 */
	NeuralNetwork() = default;

	/**
	 * @brief Constructeur de la classe NeuralNetwork.
	 *
	 * @param name Nom du réseau de neurones.
	 * @param learningRate Taux d'apprentissage du réseau.
	 * @param lossType Type de fonction de perte à utiliser.
	 * @param batchSize Taille des batch lors de l'entraînement.
	 * @param validSplit Proportion des données à utiliser pour la validation. (non utilisé)
	 * @param shuffle Indique si les données doivent être mélangées avant l'entraînement.
	 */
	NeuralNetwork(std::string name, float learningRate = 0.1, LossType lossType = mse, int batchSize = 1, float validSplit = 0.2, bool shuffle = true) {
		this->name = name;
		this->learningRate = learningRate;
		this->batchSize = batchSize;
		this->validSplit = validSplit;
		this->shuffle = shuffle;

		switch (lossType) {
			case mse:
				this->lossFunction = new MSE();
				break;

			case cross_entropy:
				this->lossFunction = new CrossEntropy();
				break;

			default:
				break;
		}
	};

	~NeuralNetwork() = default;

	/**
	 * @brief Ajoute une couche au réseau de neurones.
	 *
	 * @param layer La couche à ajouter.
	 */
	void add(ILayer *layer);

	/**
	 * @brief Effectue une itération d'entraînement sur une entrée.
	 *
	 * @param input Les données en entrée.
	 * @param label Le label des données en entrée.
	 * @param trainSize La taille de l'ensemble d'entraînement.
	 */
	void iter(xt::xarray<float> input, xt::xarray<int> label, float trainSize);

	/**
	 * @brief Met à jour les paramètres apprenables du réseau de neurones.
	 *
	 * @param size La taille du batch.
	 */
	void batch(float size);

	/**
	 * @brief Entraîne le réseau de neurones sur un ensemble de données.
	 *
	 * @param samples Les données d'entraînement.
	 * @param validSamples Les données de validation.
	 * @param epochs Le nombre d'époques d'entraînement.
	 * @param patience Le nombre d'époques à attendre avant d'arrêter l'entraînement si la perte ne diminue pas.
	 */
	void train(std::vector<std::tuple<xt::xarray<float>, xt::xarray<float>>> samples, std::vector<std::tuple<xt::xarray<float>, xt::xarray<float>>> validSamples, int epochs, int patience);

	/**
	 * @brief Évalue la performance du réseau de neurones sur un ensemble de données.
	 *
	 * @param samples Les données à évaluer.
	 * @return La précision du réseau.
	 */
	float eval(std::vector<std::tuple<xt::xarray<float>, xt::xarray<float>>> samples);

	/**
	 * @brief Charge les paramètres du réseau de neurones depuis un fichier.
	 *
	 * @param path Le chemin du fichier de sauvegarde.
	 */
	void load(const std::string path);

	/**
	 * @brief Sauvegarde les paramètres du réseau de neurones dans un fichier.
	 */
	void save() const;
};

#endif	// NETWORK_H
