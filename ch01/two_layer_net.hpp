#ifndef TWO_LAYER_NET_HPP
#define TWO_LAYER_NET_HPP

#include <vector>
#include <iostream>
#include "xtensor/xarray.hpp"
#include "xtensor/xrandom.hpp"
#include "xtensor/xio.hpp"
#include "../common/layers.hpp"
#include "../common/util.hpp"
#include "../common/model_base.hpp"

using namespace std;
using namespace xt;

class TwoLayerNet : public ModelBase
{
private:
    xarray<double> w1;
    xarray<double> b1;
    xarray<double> w2;
    xarray<double> b2;
    int I, H, O;
    SoftmaxWithLoss loss_layer;

public:
    TwoLayerNet(int input_size, int hidden_size, int output_size)
    {
        I = input_size;
        H = hidden_size;
        O = output_size;
        w1 = 0.01 * random::randn<double>({I, H});
        b1 = zeros<double>({H});
        w2 = 0.01 * random::randn<double>({H, O});
        b2 = zeros<double>({O});

        static Affine a1(w1, b1);
        static Sigmoid s;
        static Affine a2(w2, b2);

        layers.push_back(a1);
        layers.push_back(s);
        layers.push_back(a2);

        for (int i = 0; i < layers.size(); i++)
        {
            this->params.insert(this->params.end(), layers[i].get().params.begin(), layers[i].get().params.end());
            this->grads.insert(this->grads.end(), layers[i].get().grads.begin(), layers[i].get().grads.end());
        }
    }

    inline xarray<double> predict(xarray<double> &&x)
    {
        for (int i = 0; i < layers.size(); i++)
        {
            x = layers[i].get().forward(xarray<double>{x});
        }
        return x;
    }

    inline xarray<double> forward(xarray<double> &x, xarray<int> &t)
    {
        xarray<double> score = predict(std::forward<xarray<double>>(x));
        xarray<double> loss = loss_layer.forward(score, std::forward<xarray<int>>(t));
        return loss;
    }

    inline xarray<double> backward(xarray<int> &&dout = {1})
    {
        auto dx = loss_layer.backward(dout);
        for (int i = layers.size() - 1; i >= 0; i--)
        {
            dx = layers[i].get().backward(xarray<double>(dx));
        }
        return dx;
    }
};

#endif // TWO_LAYER_NET_HPP