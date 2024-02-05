#include "gtest/gtest.h"
#include "../include/conv_op.h"
#include "../include/layers/pooling.h"

#include <xtensor/xio.hpp>
#include <xtensor/xnpy.hpp>
#include <xtensor/xview.hpp>

TEST(UnitaryTest, TestRotateMatrix)
{

    xt::xarray<float> A = {{1, 2, 3},
                           {4, 5, 6},
                           {7, 8, 9}};

    xt::xarray<float> B = {{9, 8, 7},
                           {6, 5, 4},
                           {3, 2, 1}};
    rotateMatrix(A);

    ASSERT_EQ(A, B);
}

TEST(UnitaryTest, TestConvolution)
{
    xt::xarray<float> A = {{1, 2, 3},
                           {4, 5, 6},
                           {7, 8, 9}};

    xt::xarray<float> kernel = {{1, 2},
                                {3, 4}};

    xt::xarray<float> C = {{23, 33},
                           {53, 63}};

    ASSERT_EQ(matrixConvolution(A, kernel, 1, 0), C);
}

TEST(UnitaryTest, TestPadding)
{
    xt::xarray<float> A = {{1, 2, 3},
                           {4, 5, 6},
                           {7, 8, 9}};

    xt::xarray<float> B = {{0, 0, 0, 0, 0},
                           {0, 1, 2, 3, 0},
                           {0, 4, 5, 6, 0},
                           {0, 7, 8, 9, 0},
                           {0, 0, 0, 0, 0}};

    ASSERT_EQ(padMatrice(A, 1), B);
}

TEST(UnitaryTest, TestPooling)
{

    Pooling *pool_max = new Pooling{{1, 5, 5}, 2, 1, 0, PoolingType::POOLING_MAX};
    Pooling *pool_min = new Pooling{{1, 5, 5}, 2, 1, 0, PoolingType::POOLING_MIN};
    Pooling *pool_avg = new Pooling{{1, 5, 5}, 2, 1, 0, PoolingType::POOLING_AVG};

    xt::xarray<float> A = {{{1, 2, 3, 4, 5},
                            {6, 7, 8, 9, 10},
                            {11, 12, 13, 14, 15},
                            {16, 17, 18, 19, 20},
                            {21, 22, 23, 24, 25}}};

    xt::xarray<float> A_max = {{{7., 8., 9., 10.},
                                {12., 13., 14., 15.},
                                {17., 18., 19., 20.},
                                {22., 23., 24., 25.}}};

    xt::xarray<float> A_min = {{{1., 2., 3., 4.},
                                {6., 7., 8., 9.},
                                {11., 12., 13., 14.},
                                {16., 17., 18., 19.}}};

    xt::xarray<float> A_avg = {{{4., 5., 6., 7.},
                                {9., 10., 11., 12.},
                                {14., 15., 16., 17.},
                                {19., 20., 21., 22.}}};

    pool_max->forward(A);
    pool_min->forward(A);
    pool_avg->forward(A);

    ASSERT_EQ(pool_max->output,A_max);
    ASSERT_EQ(pool_min->output,A_min);
    ASSERT_EQ(pool_avg->output,A_avg);
}