#include <iostream>
#include "xtensor/xarray.hpp"
#include "xtensor/xio.hpp"
#include "xtensor/xrandom.hpp"
#include "xtensor/xview.hpp"
#include "xtensor-blas/xlinalg.hpp"

using namespace std;
using namespace xt;

template <class T>
xarray<double> sigmoid(T x) {
    return 1 / (1  + exp(-x));
}

int main() {
    // 行列の乱数を使うときには必ず必要
    random::seed(100);
    // auto演算子：型推論ができる(c++11)
    auto x = random::randn<double>({3, 2});
    auto w1 = random::randn<double>({2, 4});
    auto b1 = random::randn<double>({4});
    auto w2 = random::randn<double>({4, 3});
    auto b2 = random::randn<double>({3});

    auto h = linalg::dot(x, w1) + b1;
    cout << h << endl;
    auto a = sigmoid(h);
    cout << a << endl;
    auto s = linalg::dot(a, w2) + b2;
    cout << s << endl;
}