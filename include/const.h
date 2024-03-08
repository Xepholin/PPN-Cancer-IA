#ifndef CONST_H
#define CONST_H

// PNG = 0 / PBM sinon
const int PNGPBM = 1;

const int nbImagesTrain = 4;
const int nbImagesEval = 4;

const int PNGDim = 50;
const int PBMDim = PNGDim - 2;

#define IMAGE_TENSOR_DIM {1, PBMDim, PBMDim}

#endif