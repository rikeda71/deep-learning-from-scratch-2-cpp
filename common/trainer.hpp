#ifndef TRAINER_H //二重でincludeされることを防ぐ
#define TRAINER_H

#include <vector>
#include <tuple>
#include <string>
#include <ctime>
#include "xtensor/xarray.hpp"
#include "../common/layers.hpp"
#include "../common/optimizer.hpp"
#include "../common/util.hpp"
#include "../common/model_base.hpp"
#include "./matplotlibcpp.h"

using namespace std;
using namespace xt;
namespace plt = matplotlibcpp;

class Trainer
{
private:
    ModelBase *model;
    OptBase *optimizer;
    vector<double> loss_list;
    int eval_interval;
    int current_epoch;
    xarray<double> batch_x;
    xarray<double> epoch_x;
    xarray<int> batch_t;
    xarray<int> epoch_t;

public:
    Trainer(ModelBase *model, OptBase *optimizer)
    {
        this->model = model;
        this->optimizer = optimizer;
    }
    void fit(xarray<double> x, xarray<int> t, int max_epoch, int batch_size, int max_grad, int eval_interval);
    void plot(string ylim);
    void remove_duplicate(vector<reference_wrapper<xarray<double>>> params, vector<reference_wrapper<xarray<double>>> grads);
};

inline void Trainer::fit(xarray<double> x, xarray<int> t, int max_epoch = 10, int batch_size = 32, int max_grad = 0, int eval_interval = 20)
{
    epoch_x = xt::zeros<double>(x.shape());
    epoch_t = xt::zeros<int>(t.shape());
    const int data_size = x.shape()[0];
    const int max_iters = data_size / batch_size;
    this->eval_interval = eval_interval;
    auto model_ = this->model;
    auto optimizer_ = this->optimizer;
    xarray<double> loss;
    double total_loss = 0.0;
    double avg_loss;
    int loss_count = 0;

    int start_time = time(nullptr); //  あとでtime.hから時間計測の関数を呼ぶ
    int elapsed_time;

    for (int epoch = 0; epoch < max_epoch; epoch++)
    {
        // シャッフル
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
            loss = model_->forward(batch_x, batch_t);
            model_->backward(xarray<double>{1});
            remove_duplicate(model_->params, model_->grads);
            if (max_grad > 0.0)
            {
                clip_grads(model_->grads, max_grad);
            }
            optimizer_->update(model_->params, model_->grads); // 共有された重みを1つに集約
            total_loss = total_loss + loss(0);
            loss_count += 1;

            // 評価
            if (eval_interval > 0 and (iters % eval_interval) == 0)
            {
                elapsed_time = time(nullptr) - start_time;
                avg_loss = total_loss / loss_count;
                printf(" epoch %d |  iter %d /  %d | time %d[s] | loss %.2f\n", epoch + 1, iters + 1, max_iters, elapsed_time, avg_loss);
                loss_list.push_back(avg_loss);
                total_loss = 0;
                loss_count = 0;
            }
        }
    }
    this->current_epoch += 1;
}

inline void Trainer::plot(string ylim = "")
{
    plt::plot(this->loss_list);
    plt::xlabel("iteration (x" + to_string(this->eval_interval) + ")");
    plt::ylabel("loss");
    plt::save("./loss.png");
}

inline void Trainer::remove_duplicate(vector<reference_wrapper<xarray<double>>> params, vector<reference_wrapper<xarray<double>>> grads)
{
    /*
        パラメータ配列中の重複する重みを1つに集約し，
        その重みに対応する勾配を加算する
    */
    while (true)
    {
        bool find_flg = false;
        int L = params.size();

        for (int i = 0; i < L - 1; i++)
        {
            for (int j = i + 1; j < L; j++)
            {
                // 重みを共有する場合
                if (params[i].get() == params[j].get())
                {
                    grads[i].get() += grads[j].get(); // 勾配の加算
                    find_flg = true;
                    params.erase(params.begin() + j);
                    grads.erase(grads.begin() + j);
                }
                // 転置行列として重みを共有する場合
                else if (
                    params[i].get().dimension() == 2 &&
                    params[j].get().dimension() == 2 &&
                    transpose(params[i].get()).shape() == params[j].get().shape() &&
                    (transpose(params[i].get()) == params[j].get()))
                {
                    grads[i].get() += transpose(grads[j].get());
                    find_flg = true;
                    params.erase(params.begin() + j);
                    grads.erase(grads.begin() + j);
                }
                if (find_flg)
                {
                    break;
                }
                if (find_flg)
                {
                    break;
                }
            }
        }
        if (!find_flg)
        {
            break;
        }
    }
}

#endif