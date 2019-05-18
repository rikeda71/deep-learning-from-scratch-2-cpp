#ifndef UTIL_H //二重でincludeされることを防ぐ
#define UTIL_H

#include <iostream>
#include <vector>
#include "xtensor/xarray.hpp"

using namespace std;

vector<vector<int>> to_indexes(xt::xarray<int> t)
{
    int batch_size = t.shape()[0];
    vector<vector<int>> indexes(batch_size);
    for (int i = 0; i < batch_size; i++)
    {
        indexes[i] = {i, t[i]};
    }
    return indexes;
}

template <class T>
void cout_vector(T vec)
{
    cout << "{ ";
    for (int i = 0; i < vec.size(); i++)
    {
        if (i > 0)
        {
            cout << ", ";
        }
        cout << vec.at(i);
    }
    cout << " }" << endl;
}

template <typename T>
vector<vector<T>> xarray2vector(xt::xarray<T> arr)
{
    int row = arr.shape()[0];
    int column = arr.shape()[1];
    vector<vector<T>> vec(row);
    for (int i = 0; i < row; i++)
    {
        vec[i].resize(column);
    }
    for (int i = 0; i < row; i++)
    {
        for (int j = 0; j < column; j++)
        {
            vec[i][j] = arr(i, j);
        }
    }
    return vec;
}

#endif