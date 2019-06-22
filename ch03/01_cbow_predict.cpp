#include <iostream>
#include "xtensor/xarray.hpp"
#include "xtensor/xio.hpp"
#include "xtensor/xrandom.hpp"
#include "../common/layers.hpp"

int main()
{
    // サンプルのコンテキストデータ
    auto c0 = xt::xarray<int>{{1, 0, 0, 0, 0, 0, 0}};
    auto c1 = xt::xarray<int>{{0, 0, 1, 0, 0, 0, 0}};

    // 重みの初期化
    auto W_in = xt::random::randn<double>({7, 3});
    auto W_out = xt::random::randn<double>({3, 7});

    // レイヤの生成
    MatMul in_layer0(W_in);
    MatMul in_layer1(W_in);
    MatMul out_layer(W_out);

    // 順伝播
    auto h0 = in_layer0.forward(c0);
    auto h1 = in_layer1.forward(c1);
    auto h = 0.5 * (h0 + h1);
    auto s = out_layer.forward(h);

    std::cout << s << std::endl;
}