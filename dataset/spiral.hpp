#ifndef SPIRAL_HPP
#define SPIRAL_HPP

#include <tuple>
#include <vector>
#include "xtensor/xarray.hpp"
#include "xtensor/xrandom.hpp"
#include "../common/util.hpp"

std::tuple<xt::xarray<double>, xt::xarray<int>> load_data(int seed = 1984)
{
    xt::random::seed(seed);
    const int N = 100;     // sample size
    const int DIM = 2;     // dimension size
    const int CLS_NUM = 3; // class size
    double rate, radius, theta;
    xt::xarray<double> randn = xt::random::randn<double>({CLS_NUM, N});
    int ix;

    xt::xarray<double> x = xt::zeros<double>({N * CLS_NUM, DIM});
    xt::xarray<int> t = xt::zeros<int>({N * CLS_NUM, CLS_NUM});

    for (int j = 0; j < CLS_NUM; j++)
    {
        for (int i = 0; i < N; i++)
        {
            rate = 1.0 * i / N;
            radius = 1.0 * rate;
            theta = j * 4.0 + 4.0 * rate + randn[j, i] * 0.2;

            ix = N * j + i;
            x(ix, 0) = radius * std::sin(theta);
            x(ix, 1) = radius * std::cos(theta);
            t(ix, j) = 1;
        }
    }

    return {x, t};
}

#endif // SPIRARL_HPP