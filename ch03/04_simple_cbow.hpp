#ifndef SIMPLE_CBOW
#define SIMPLE_CBOW

#include <vector>
#include "xtensor/xarray.hpp"
#include "xtensor/xio.hpp"
#include "xtensor/xrandom.hpp"
#include "../common/model_base.hpp"

using namespace std;
using namespace xt;

class SimpleCBOW : public ModelBase
{
private:
    xarray<double> W_in;
    xarray<double> W_out;
    xarray<double> w2;
    xarray<double> b2;
    SoftmaxWithLoss loss_layer;

public:
    xarray<double> *word_vecs;

    SimpleCBOW(int vocab_size, int hidden_size)
    {
        const int V = vocab_size;
        const int H = hidden_size;

        W_in = 0.01 * random::randn<double>({V, H});
        W_out = 0.01 * random::randn<double>({H, V});

        static MatMul in_layer0(W_in);
        static MatMul in_layer1(W_in);
        static MatMul out_layer(W_out);

        layers.push_back(in_layer0);
        layers.push_back(in_layer1);
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
        auto h0 = this->layers[0].get().forward(xt::view(contexts, xt::all(), 0));
        auto h1 = this->layers[1].get().forward(xt::view(contexts, xt::all(), 1));
        auto h = (h0 + h1) * 0.5;
        auto score = this->layers[2].get().forward(h);
        auto loss = this->loss_layer.forward(score, std::forward<xarray<int>>(target));
        return loss;
    }

    inline xarray<double> backward(xarray<int> &&dout = {1})
    {
        auto ds = this->loss_layer.backward(dout);
        auto da = this->layers[2].get().backward(std::forward<xarray<double>>(ds));
        da = da * 0.5;
        auto dx = this->layers[1].get().backward(xarray<double>{da});
        dx += this->layers[0].get().backward(xarray<double>{da});
        return dx;
    }
};

#endif