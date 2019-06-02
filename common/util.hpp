#ifndef UTIL_H //二重でincludeされることを防ぐ
#define UTIL_H

#include <iostream>
#include <vector>
#include <tuple>
#include <unordered_map>
#include <string>
#include <sstream>
#include "xtensor/xarray.hpp"
#include "xtensor/xadapt.hpp"
#include "xtensor/xmath.hpp"
#include "xtensor/xutils.hpp"
#include "xtensor/xio.hpp"
#include "xtensor-blas/xlinalg.hpp"

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

void clip_grads(vector<reference_wrapper<xt::xarray<double>>> grads, double max_norm)
{
    double total_norm = 0;
    for (int i = 0; i < grads.size(); i++)
    {
        total_norm += xt::sum(xt::square(grads[i].get()))(0);
        // auto tmp_square = xt::square(grads[i].get());
        // auto tmp_sum = xt::sum(tmp_square);
        // total_norm += tmp_sum(0);
    }
    total_norm = std::sqrt(total_norm);

    double rate = max_norm / (total_norm + 1e-6);
    if (rate < 1)
    {
        for (int i = 0; i < grads.size(); i++)
        {
            grads[i].get() *= rate;
        }
    }
}

vector<string> split(string str, char deliminator)
{
    vector<string> elems;
    stringstream ss{str};
    string split_s;
    cout << str << endl;
    while (getline(ss, split_s, deliminator))
    {
        if (!split_s.empty())
        {
            elems.push_back(split_s);
        }
    }
    return elems;
}

string replace_string(string str, string target, string replacement) {
    if (target.empty()) return str;
    string::size_type pos = 0;
    while((pos = str.find(target, pos)) != string::npos) {
        str.replace(pos, str.length(), replacement);
        pos += replacement.length();
    }
    return str;
}

tuple<xt::xarray<int>, unordered_map<string, int>, vector<string>> preprocess(string text)
{
    int new_id;
    string word;
    vector<int> ids;
    // tolower
    transform(text.begin(), text.end(), text.begin(), ::tolower);
    // replace "." -> " ."
    text = replace_string(text, ".", " .");
    vector<string> words = split(text, ' ');

    unordered_map<string, int> word_to_id;
    vector<string> id_to_word;

    for (int i = 0; i < words.size(); i++)
    {
        word = words[i];
        if (word_to_id.count(word) == 0)
        {
            new_id = word_to_id.size();
            word_to_id[word] = new_id;
            id_to_word.push_back(word);
            ids.push_back(new_id);
        }
        else
        {
            ids.push_back(word_to_id[word]);
        }
    }

    std::vector<std::size_t> shape = {words.size()};
    auto corpus = xt::adapt(ids, shape);
    return {corpus, word_to_id, id_to_word};
}

#endif