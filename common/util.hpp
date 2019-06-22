#ifndef UTIL_H //二重でincludeされることを防ぐ
#define UTIL_H

#include <iostream>
#include <vector>
#include <tuple>
#include <unordered_map>
#include <string>
#include <sstream>
#include <algorithm>
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

string replace_string(string str, string target, string replacement)
{
    if (target.empty())
        return str;
    string::size_type pos = 0;
    while ((pos = str.find(target, pos)) != string::npos)
    {
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

xt::xarray<int> create_co_matrix(xt::xarray<int> corpus, int vocab_size, int window_size = 1)
{
    int word_id;
    int left_idx;
    int right_idx;
    int left_word_id;
    int right_word_id;
    int corpus_size = corpus.size();
    xt::xarray<int> co_matrix = xt::zeros<int>({vocab_size, vocab_size});

    for (int idx = 0; idx < corpus_size; idx++)
    {
        for (int i = 1; i < window_size + 1; i++)
        {
            word_id = corpus(idx);
            left_idx = idx - i;
            right_idx = idx + i;

            if (left_idx >= 0)
            {
                left_word_id = corpus(left_idx);
                co_matrix(word_id, left_word_id) = co_matrix(word_id, left_word_id) + 1;
            }
            if (right_idx < corpus_size)
            {
                right_word_id = corpus(right_idx);
                co_matrix(word_id, right_word_id) = co_matrix(word_id, right_word_id) + 1;
            }
        }
    }
    return co_matrix;
}

xt::xarray<double> cos_similarity(xt::xarray<double> &&x, xt::xarray<double> &&y, double eps = 1e-8)
{
    auto nx = xt::xarray<double>{x} / xt::sqrt(xt::sum(xt::pow(xt::xarray<double>{x}, 2)));
    auto ny = xt::xarray<double>{y} / xt::sqrt(xt::sum(xt::pow(xt::xarray<double>{y}, 2)));
    return xt::linalg::dot(nx, ny);
}

void most_similar(string query, unordered_map<string, int> word_to_id, vector<string> id_to_word, xt::xarray<double> word_matrix, int top = 5)
{
    // クエリを取り出す
    if (word_to_id.count(query) == 0)  {
        cout << query << " is not found" << endl;
        return;
    }

    cout << "\n [query] " << query << endl;
    int query_id = word_to_id[query];
    auto query_vec = xt::view(word_matrix, query_id, xt::all());

    // cos類似度の算出
    int vocab_size = id_to_word.size();
    vector<double> similarity;
    for (int i =  0; i < vocab_size; i++) {
        similarity.push_back(cos_similarity(xt::view(word_matrix, i, xt::all()), query_vec)[0]);
    }

    // cos類似度の結果から，その値を高い順に出力
    std::sort(
        similarity.begin(),
        similarity.end(),
        greater<double>()
    );
    int i = 0;
    int cnt = 0;
    while(1) {
        if (cnt >= top || i >= similarity.size()) return;
        if (id_to_word[i]  == query) {
            i++;
            continue;
        }
        cout << id_to_word[i] << ": " << similarity[i]  << endl;
        i++;
        cnt++;
    }
}

#endif