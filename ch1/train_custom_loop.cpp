#include <vector>
#include "xtensor/xarray.hpp"
#include "xtensor/xrandom.hpp"
#include "xtensor/xview.hpp"
#include "xtensor/xio.hpp"
#include "../common/optimizer.hpp"
#include "../dataset/spiral.hpp"
#include "matplotlibcpp.h"
#include "two_layer_net.hpp"

#include <typeinfo>

using namespace std;
using namespace xt;
namespace plt = matplotlibcpp;

int main()
{
    // ハイパーパラメータの設定
    int max_epoch = 300;
    int batch_size = 30;
    int hidden_size = 10;
    double learning_rate = 1.0;

    // データの読み込み，モデルとオプティマイザの生成
    auto [x, t] = load_data();
    auto model = TwoLayerNet(2, hidden_size, 3);
    auto optimizer = SGD(learning_rate);

    // 学習で使用する変数
    int data_size = x.shape()[0];
    int max_iters = data_size / batch_size;
    int loss_count = 0;
    double total_loss = 0;
    double avg_loss;
    xarray<double> loss;
    vector<double> loss_list;
    xarray<double> batch_x;
    xarray<int> batch_t;
    xarray<double> epoch_x = xt::zeros<double>(x.shape());
    xarray<int> epoch_t = xt::zeros<int>(t.shape());

    xt::random::seed(100);

    for (int epoch = 0; epoch < max_epoch; epoch++)
    {
        // データのシャッフル
        auto idx = xt::random::permutation<int>(data_size);
        for (int i = 0; i < data_size; i++)
        {
            for (int j = 0; j < 3; j++)
            {
                epoch_x(i, j) = x(idx(i), j);
                epoch_t(i, j) = t(idx(i), j);
            }
        }

        for (int iters = 0; iters < max_iters; iters++)
        {
            batch_x = xt::view(epoch_x, xt::range(iters * batch_size, (iters + 1) * batch_size), xt::all());
            batch_t = xt::view(epoch_t, xt::range(iters * batch_size, (iters + 1) * batch_size), xt::all());

            // 勾配を求め，パラメータを更新
            loss = model.forward(batch_x, batch_t);
            model.backward();
            optimizer.update(model.params, model.grads);

            total_loss = total_loss + loss(0);
            loss_count += 1;

            // 定期的に学習経過を出力
            if ((iters + 1) % 10 == 0)
            {
                avg_loss = total_loss / loss_count;
                printf(" epoch %d |  iter %d /  %d |  loss %.2f\n", epoch + 1, iters + 1, max_iters, avg_loss);
                loss_list.push_back(avg_loss);
                total_loss = 0;
                loss_count = 0;
            }
        }
    }
}