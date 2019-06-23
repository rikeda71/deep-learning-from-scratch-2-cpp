#ifndef SIMPLE_SKIPGRAM
#define SIMPLE_SKIPGRAM

#include <vector>
#include "xtensor/xarray.hpp"
#include "xtensor/xio.hpp"
#include "xtensor/xrandom.hpp"
#include "../common/model_base.hpp"

using namespace std;
using namespace xt;

class SimpleSkipGram : public ModelBase
{
private:
    xarray<double> W_in;
    xarray<double> W_out;
    xarray<double> w2;
    xarray<double> b2;
    SoftmaxWithLoss loss_layer1;
    SoftmaxWithLoss loss_layer2;

public:
    xarray<double> *word_vecs;

    SimpleSkipGram(int vocab_size, int hidden_size)
    {
        const int V = vocab_size;
        const int H = hidden_size;

        W_in = 0.01 * random::randn<double>({V, H});
        W_out = 0.01 * random::randn<double>({H, V});

        static MatMul in_layer(W_in);
        static MatMul out_layer(W_out);

        layers.push_back(in_layer);
        layers.push_back(out_layer);

        for (int i = 0; i < layers.size(); i++)
        {
            this->params.insert(this->params.end(), layers[i].get().params.begin(), layers[i].get().params.end());
            this->grads.insert(this->grads.end(), layers[i].get().grads.begin(), layers[i].get().grads.end());
        }

        word_vecs = &W_in;
    }

    inline xarray<double> predict(xarray<double> &&x)
    {
        // don't use function
        return x;
    }

    inline xarray<double> forward(xarray<double> &x, xarray<int> &t)
    {
        auto contexts = std::forward<xarray<double>>(x);
        auto target = std::forward<xarray<int>>(t);
        auto h = this->layers[0].get().forward(std::forward<xarray<double>>(target));
        auto s = this->layers[1].get().forward(std::forward<xarray<double>>(h));
        auto l1 = this->loss_layer1.forward(s, std::forward<xarray<int>>(xt::view(contexts, xt::all(), 0)));
        auto l2 = this->loss_layer2.forward(s, std::forward<xarray<int>>(xt::view(contexts, xt::all(), 1)));
        auto loss = l1 + l2;
        return loss;
    }

    inline xarray<double> backward(xarray<int> &&dout = {1})
    {
        auto dl1 = this->loss_layer1.backward(dout);
        auto dl2 = this->loss_layer2.backward(dout);
        auto ds = dl1 + dl2;
        auto dh = this->layers[1].get().backward(std::forward<xarray<double>>(ds));
        auto dx = this->layers[0].get().backward(xarray<double>{dh});
        return dx;
    }
};

#endif