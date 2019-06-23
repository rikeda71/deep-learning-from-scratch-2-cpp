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
#include "xtensor/xadapt.hpp"
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
    auto nx = xt::xarray<double>{x} / xt::sqrt(xt::sum(xt::pow(xt::xarray<double>{x}, 2)) + eps);
    auto ny = xt::xarray<double>{y} / xt::sqrt(xt::sum(xt::pow(xt::xarray<double>{y}, 2)) + eps);
    return xt::linalg::dot(nx, ny);
}

void most_similar(string query, unordered_map<string, int> word_to_id, vector<string> id_to_word, xt::xarray<double> word_matrix, int top = 5)
{
    // クエリを取り出す
    if (word_to_id.count(query) == 0)
    {
        cout << query << " is not found" << endl;
        return;
    }

    cout << "\n [query] " << query << endl;
    int query_id = word_to_id[query];
    auto query_vec = xt::view(word_matrix, query_id, xt::all());

    // cos類似度の算出
    int vocab_size = id_to_word.size();
    vector<double> similarity;
    for (int i = 0; i < vocab_size; i++)
    {
        similarity.push_back(cos_similarity(xt::view(word_matrix, i, xt::all()), query_vec)[0]);
    }

    // cos類似度の結果から，その値を高い順に出力
    std::sort(
        similarity.begin(),
        similarity.end(),
        greater<double>());
    int i = 0;
    int cnt = 0;
    while (1)
    {
        if (cnt >= top || i >= similarity.size())
            return;
        if (id_to_word[i] == query)
        {
            i++;
            continue;
        }
        cout << id_to_word[i] << ": " << similarity[i] << endl;
        i++;
        cnt++;
    }
}

xt::xarray<double> ppmi(xt::xarray<int> C, bool verbose = false, double eps = 1e-8)
{
    auto shape = C.shape();
    xt::xarray<double> M = xt::zeros<double>({shape[0], shape[1]});
    xt::xarray<double> N = xt::sum(C);
    xt::xarray<double> S = xt::sum(C, 0);
    int total = shape[0] * shape[1];
    int cnt = 0;
    double pmi;

    for (int i = 0; i < shape[0]; i++)
    {
        for (int j = 0; j < shape[1]; j++)
        {
            pmi = xt::log2(C(i, j) * N / (S(j) * S(i)) + eps)[0];
            M(i, j) = (0.0 < pmi) ? pmi : 0.0;

            if (verbose)
            {
                cnt += 1;
                if (cnt % (total / 100) == 0)
                {
                    cout << 100 * cnt / total << "% done" << endl;
                }
            }
        }
    }
    return M;
}

tuple<xt::xarray<int>, xt::xarray<int>> create_contexts_target(xt::xarray<int> corpus, int window_size = 1)
{
    xt::xarray<int> target = xt::view(corpus, xt::range(window_size, corpus.size() - window_size));
    vector<int> contexts;

    unsigned int cnt = 0; // コンテキストのループを通った回数をカウント
    for (int idx = window_size; idx < corpus.size() - window_size; idx++)
    {
        for (int t = -window_size; t < window_size + 1; t++)
        {
            if (t == 0)
                continue;
            contexts.push_back(corpus(idx + t));
        }
        cnt++;
    }
    vector<std::size_t> shape{cnt, (unsigned int)window_size * 2};
    xt::xarray<int> contexts_vector = xt::adapt(contexts, shape);
    return {contexts_vector, target};
}

xt::xarray<int> convert_one_hot(xt::xarray<int> corpus, int vocab_size)
{

    int N = corpus.shape()[0];
    int C;
    int word_id;
    xt::xarray<int> one_hot;

    if (corpus.dimension() == 1)
    {
        one_hot = xt::zeros<int>({N, vocab_size});
        for (int idx = 0; idx < N; idx++)
        {
            word_id = corpus(idx);
            xt::view(one_hot, idx, word_id) = 1;
        }
    }
    else if (corpus.dimension() == 2)
    {
        C = corpus.shape()[1];
        one_hot = xt::zeros<int>({N, C, vocab_size});
        for (int idx_0 = 0; idx_0 < N; idx_0++)
        {
            auto word_ids = xt::view(corpus, idx_0, xt::all());
            for (int idx_1 = 0; idx_1 < word_ids.size(); idx_1++)
            {
                word_id = corpus(idx_0, idx_1);
                xt::view(one_hot, idx_0, idx_1, word_id) = 1;
            }
        }
    }

    return one_hot;
}

#endif