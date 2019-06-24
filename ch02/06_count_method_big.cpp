#include <iostream>
#include <string>
#include <vector>
#include "xtensor/xarray.hpp"
#include "xtensor/xio.hpp"
#include "xtensor-blas/xlinalg.hpp"
#include "../common/util.hpp"
#include "../dataset/ptb.hpp"
#include "matplotlibcpp.h"

using namespace std;

int main()
{
    const int window_size = 2;
    const int wordvec_size = 100;
    auto [corpus, w2i, i2w] = ptb::load_data("train");
    int vocab_size = w2i.size();
    auto C = create_co_matrix(corpus, vocab_size, window_size);
    cout << "calculating PPMI ..." << endl;
    xt::xarray<double> W = ppmi(C);

    cout << "calculating SVD ..." << endl;
    auto [U, S, V] = xt::linalg::svd(W);

    xt::xarray<double> word_vecs = xt::view(U, xt::all(), xt::range(0, wordvec_size));
    vector<string> querys = {"you", "year", "car", "toyota"};
    string query;

    for (int i = 0; i < word_vecs.size(); i++)
    {
        query = querys[i];
        most_similar(query, w2i, i2w, word_vecs, 5);
    }
}