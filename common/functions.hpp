#ifndef FUNC_H
#define FUNC_H

#include <vector>
#include "xtensor/xarray.hpp"
#include "xtensor/xsort.hpp"
#include "xtensor/xindex_view.hpp"
#include "xtensor/xio.hpp"
#include "util.hpp"

using namespace xt;

template <typename T>
T sigmoid(T &&x)
{
    return 1 / (1 + exp(-std::forward<T>(x)));
}

template <typename T>
T relu(T &&x)
{
    return maximum(0, std::forward<T>(x));
}

xarray<double> softmax(xarray<double> &&x)
{
    xarray<double> amax_tmp;
    xarray<double> sum_tmp;
    if (x.dimension() == 2)
    {
        // reshapeしないとbroadcast errorが起こる
        amax_tmp = amax(xarray<double>{x}, {1});
        amax_tmp.reshape({-1, 1});
        x = xarray<double>{x} - amax_tmp;
        x = exp(xarray<double>{x});
        sum_tmp = sum(xarray<double>{x}, {1});
        sum_tmp.reshape({-1, 1});
        x = xarray<double>{x} / sum_tmp;
    }
    else if (x.dimension() == 1)
    {
        amax_tmp = amax(xarray<double>{x});
        x = xarray<double>{x} - amax_tmp;
        sum_tmp = sum(xarray<double>{x});
        x = exp(xarray<double>{x}) / sum_tmp;
    }
    return x;
}

template <typename T>
T cross_entropy_error(T &&y, xarray<int> &&t)
{
    if (y.dimension() == 1)
    {
        t.reshape({1, -1});
        y.reshape({1, -1});
    }

    // 教師データがone-hot-vectorの場合，正解ラベルのインデックスに変換
    if (t.size() == y.size())
    {
        t = argmax(xarray<int>{t}, 1);
    }

    int batch_size = y.shape()[0];
    vector<vector<int>> indexes = to_indexes(xarray<int>{t});
    xarray<double> antilog = index_view(y, indexes);

    return -sum(log(antilog), {0}) / batch_size;
}

#endif