#ifndef IMAGE_H
#define IMAGE_H

#include <xtensor/xarray.hpp>
#include <xtensor/xrandom.hpp>

#include "const.h"

/**
 * @brief Classe représentant une image.
 */
class Image {
   public:
	// Matrices with size 50x50
	xt::xarray<uint8_t> r = xt::empty<uint8_t>({PNGDim, PNGDim});  // Matrice pour les composantes rouges de l'image.
	xt::xarray<uint8_t> g = xt::empty<uint8_t>({PNGDim, PNGDim});  // Matrice pour les composantes vertes de l'image.
	xt::xarray<uint8_t> b = xt::empty<uint8_t>({PNGDim, PNGDim});  // Matrice pour les composantes bleues de l'image.

	/**
	 * @brief Enregistre l'image au format PNG.
	 *
	 * @param outputPath Le chemin de sortie du fichier PNG.
	 */
	void saveToPNG(const char *outputPath);

	/**
	 * @brief Convertit l'image en tenseur.
	 *
	 * @return Le tenseur représentant l'image.
	 */
	xt::xarray<float> toTensor();
};

/**
 * @brief Importe une image depuis un fichier.
 *
 * @param filename Le chemin du fichier à importer.
 * @return L'image importée.
 */
Image importImage(const char *filename);

/**
 * @brief Lit un fichier binaire contenant les composantes d'une image et les ajoute à une image existante.
 *
 * @param filename Le chemin du fichier à lire.
 * @param a L'image existante à laquelle ajouter les composantes lues.
 * @return L'image avec les composantes ajoutées.
 */
Image readByteFile(const char *filename, Image a);

/**
 * @brief Convertit une image RGB en niveaux de gris.
 *
 * @param a L'image à convertir.
 * @return L'image convertie en niveaux de gris.
 */
xt::xarray<float> toGrayScale(Image a);

/**
 * @brief Applique le filtre de Sobel à une image en niveaux de gris.
 *
 * @param grayMatrice L'image en niveaux de gris.
 * @return L'image des contours détectés par l'opérateur de Sobel.
 */
xt::xarray<bool> toSobel(xt::xarray<float> grayMatrice);

/**
 * @brief Applique un flou gaussien à une image.
 *
 * @param image L'image à flouter.
 * @param radius L'intensité du flou gaussien.
 * @return L'image floutée.
 */
xt::xarray<float> gaussianBlur(xt::xarray<float> image, int radius);

/**
 * @brief Enregistre une image en niveaux de gris dans un fichier PNG.
 *
 * @param outputPath Le chemin de sortie du fichier PNG.
 * @param grayMatrice L'image en niveaux de gris à enregistrer.
 */
void saveGrayToPNG(const char *outputPath, xt::xarray<uint8_t> grayMatrice);

/**
 * @brief Enregistre une image binaire au format PBM (Portable BitMap).
 *
 * @param outputPath Le chemin de sortie du fichier PBM.
 * @param boolMatrice L'image binaire à enregistrer.
 */
void saveEdgetoPBM(const char *outputPath, xt::xarray<bool> boolMatrice);

/**
 * @brief Génère des images PBM à partir d'un dossier contenant des images PNG.
 *
 * @param folderConvPath Le chemin du dossier contenant les images PNG à convertir.
 * @param folderOutput Le chemin du dossier de sortie où enregistrer les images PBM.
 */
void generateAllPBM(const char *folderConvPath, const char *folderOutput);

/**
 * @brief Importe une image au format PBM (Portable BitMap).
 *
 * @param path Le chemin du fichier PBM à importer.
 * @return L'image importée.
 */
xt::xarray<bool> importPBM(const char *path);

/**
 * @brief Importe plusieurs images au format PBM (Portable BitMap).
 *
 * @param path Le chemin du dossier contenant les fichiers PBM à importer.
 * @param nbPBM Le nombre d'images à importer.
 * @return Un tenseur contenant les images importées.
 */
xt::xarray<bool> importAllPBM(const char *path, int nbPBM);

/**
 * @brief Importe plusieurs images au format PNG.
 *
 * @param path Le chemin du dossier contenant les fichiers PNG à importer.
 * @param nbPNG Le nombre d'images à importer.
 * @return Un tenseur contenant les images importées.
 */
xt::xarray<float> importAllPNG(const char *path, int nbPNG);

#endif	// IMAGE_H
