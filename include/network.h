#ifndef NETWORK_H
#define NETWORK_H

#include <string>

#include <xtensor/xarray.hpp>
#include <xtensor/xio.hpp>
#include <xtensor/xview.hpp>

xt::xarray<bool> importPBM(const char *path);
xt::xarray<bool> importAllPBM(const char *path, int nbPBM);

#endif