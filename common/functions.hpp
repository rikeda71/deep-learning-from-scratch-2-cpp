#ifndef FUNC_H
#define FUNC_H

#include "xtensor/xarray.hpp"
#include "xtensor/xsort.hpp"

using namespace xt;

template <typename T>
T sigmoid(T x)
{
    return 1 / (1 + exp(-x));
}

template <typename T>
T relu(T x)
{
    return maximum(0, x);
}

template <typename T>
T softmax(T x)
{
    if (x.dimension() == 2)
    {
        x = x - amax(x, {1});
        x = exp(x);
        x = x / sum(x, {1});
    }
    else if (x.dimension() == 1)
    {
        x = x - amax(x);
        x = exp(x) / sum(exp(x));
    }
}

template <typename T>
T cross_entropy_error(T y, xarray<int> t)
{
    if (y.dimension() == 1)
    {
        t.reshape({1, -1});
        y.reshape({1, -1});
    }

    // 教師データがone-hot-vectorの場合，正解ラベルのインデックスに変換
    if (t.size() == y.size())
    {
        t = argmax(t, 1);
    }

    int batch_size = y.shape()[0];
    xarray<double> antilog = y[arange(batch_size), t] + 1e-7;
    return -sum(log(antilog), {0}) / batch_size;
}

#endif