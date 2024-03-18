#ifndef CONST_H
#define CONST_H

// PNG = 0 / PBM sinon
const int PNGPBM = 0;

const int nbImagesTrain = 150000;
const int nbImagesEval = 7000;

const int PNGDim = 50;
const int PBMDim = PNGDim - 2;

#define IMAGE_TENSOR_DIM {3, PNGDim, PNGDim}

#endif