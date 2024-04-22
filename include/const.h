#ifndef CONST_H
#define CONST_H

#include <iostream>
#include <tuple>
#include <string>

// PNG = 0 / PBM sinon
const int PNGPBM = 1;

const int nbImagesTrain = 150000;
const int nbImagesEval = 7000;

const int PNGDim = 50;
const int PBMDim = PNGDim - 2;

const std::string pathPNG = "../assets/breast/";
const std::string pathPBM = "../../processed_red_36/";

const std::string trainPathPNG = pathPNG + "train";
const std::string evalPathPNG = pathPNG + "eval";

const std::string trainPathPBM = pathPBM + "train";
const std::string evalPathPBM = pathPBM + "eval";

const std::string savePath = "../saves";

// #define IMAGE_TENSOR_DIM {3, PNGDim, PNGDim}
#define IMAGE_TENSOR_DIM {1, PBMDim, PBMDim}

#endif