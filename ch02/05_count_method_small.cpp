#include <iostream>
#include <string>
#include <vector>
#include "xtensor/xarray.hpp"
#include "xtensor/xio.hpp"
#include "xtensor-blas/xlinalg.hpp"
#include "../common/util.hpp"
#include "matplotlibcpp.h"

using namespace std;
namespace plt = matplotlibcpp;

int main()
{
    string text = "You say goodbye and I say hello.";
    auto [corpus, w2i, i2w] = preprocess(text);
    int vocab_size = w2i.size();
    auto C = create_co_matrix(corpus, vocab_size);
    xt::xarray<double> W = ppmi(C);
    auto [U, S, V] = xt::linalg::svd(W);

    cout << "共起行列" << endl
         << xt::view(C, 0, xt::all()) << endl;
    cout << "PPMI行列" << endl
         << xt::view(W, 0, xt::all()) << endl;
    cout << "SVD" << endl
         << xt::view(U, 0, xt::all()) << endl;
    cout << xt::view(U, 0, xt::range(0, 2)) << endl;

    vector<double> x;
    vector<double> y;
    auto U_shape = U.shape();
    for (int i = 0; i < U_shape[0]; i++)
    {
        x.push_back(U(i, 0));
        y.push_back(U(i, 1));
    }
    plt::scatter(x, y, 10);

    string word;
    int word_id;
    for (auto item = w2i.begin(); item != w2i.end(); ++item)
    {
        word = item->first;
        word_id = item->second;
        plt::annotate(word, U(word_id, 0), U(word_id, 1));
    }
    plt::save("image.png");
}