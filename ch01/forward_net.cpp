#include <iostream>
#include <vector>
#include "xtensor/xarray.hpp"
#include "xtensor/xio.hpp"
#include "xtensor/xrandom.hpp"
#include "xtensor-blas/xlinalg.hpp"

using namespace std;
using namespace xt;

class Base
{
  public:
    vector<xarray<double>> params;
    // 基底クラスを定義するときは`~Class名() {}`を定義する必要がある
    virtual ~Base() {}
    virtual xarray<double> forward(xarray<double> x) {}
};

class Sigmoid : public Base
{
  public:
    Sigmoid() {}
    xarray<double> forward(xarray<double> x)
    {
        return 1 / (1 + exp(-std::forward<xarray<double>>(x)));
    }
};

class Affine : public Base
{
  public:
    Affine(xarray<double> w, xarray<double> b)
    {
        params.push_back(w);
        params.push_back(b);
    }

    xarray<double> forward(xarray<double> x)
    {
        xarray<double> w = params[0];
        xarray<double> b = params[1];
        xarray<double> out = linalg::dot(std::forward<xarray<double>>(x), w) + b;
        return out;
    }
};

class TwoLayerNet
{
  private:
    xarray<double> w1;
    xarray<double> b1;
    xarray<double> w2;
    xarray<double> b2;

  public:
    Base *layers[3];
    xarray<double> *params[4];

    TwoLayerNet(int input_size, int hidden_size, int output_size)
    {
        const int I = input_size;
        const int H = hidden_size;
        const int O = output_size;

        w1 = random::randn<double>({I, H});
        b1 = random::randn<double>({H});
        w2 = random::randn<double>({H, O});
        b2 = random::randn<double>({O});

        static Affine a1(w1, b1);
        static Sigmoid s;
        static Affine a2(w2, b2);

        // 派生クラスのアドレスを基底クラスの配列に渡すことで
        // 派生クラスの配列を持つことができる
        layers[0] = &a1;
        layers[1] = &s;
        layers[2] = &a2;

        int cnt = 0;
        for (int i = 0; i < 3; i++)
        {
            for (int j = 0; j < layers[i]->params.size(); j++)
            {
                // 各層のパラメータのポインタを持っておくことで
                // 層内のパラメータの値が更新されても対応できる
                params[cnt] = &layers[i]->params[j];
                cnt++;
            }
        }
    }

    xarray<double> predict(xarray<double> x)
    {
        for (int i = 0; i < 3; i++)
        {
            x = layers[i]->forward(x);
        }
        return x;
    }
};

int main()
{
    random::seed(100);
    xarray<double> x = random::randn<double>({10, 2});
    TwoLayerNet model(2, 4, 3);
    xarray<double> s = model.predict(x);
    cout << "predicted" << endl
         << s << endl;
}