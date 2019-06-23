#ifndef LAYERS_H
#define LAYERS_H

#include <vector>
#include "xtensor/xarray.hpp"
#include "xtensor/xrandom.hpp"
#include "xtensor/xsort.hpp"
#include "xtensor-blas/xlinalg.hpp"
#include "functions.hpp"

#include "xtensor/xio.hpp"
#include "util.hpp"

using namespace std;
using namespace xt;

// 基底クラス
class Base
{
public:
  vector<xarray<double>> params, grads;
  // 基底クラスを定義するときは`~Class名() {}`を定義する必要がある
  virtual ~Base() {}
  virtual xarray<double> forward(xarray<double> &&x) = 0;
  virtual xarray<double> backward(xarray<double> &&dout) = 0;
};

class Sigmoid : public Base
{
public:
  Sigmoid() {}
  virtual xarray<double> forward(xarray<double> &&x);
  virtual xarray<double> backward(xarray<double> &&dout);

private:
  xarray<double> out;
};

inline xarray<double> Sigmoid::forward(xarray<double> &&x)
{
  auto out = 1 / (1 + exp(-xarray<double>{x}));
  this->out = out;
  return out;
}

inline xarray<double> Sigmoid::backward(xarray<double> &&dout)
{
  xarray<double> dx = std::forward<xarray<double>>(dout) * (1.0 - this->out) * this->out;
  return dx;
}

class Softmax : public Base
{
public:
  Softmax() {}
  virtual xarray<double> forward(xarray<double> &&x);
  virtual xarray<double> backward(xarray<double> &&dout);

private:
  xarray<double> out;
};

inline xarray<double> Softmax::forward(xarray<double> &&x)
{
  this->out = softmax(std::forward<xarray<double>>(x));
  return this->out;
}

inline xarray<double> Softmax::backward(xarray<double> &&dout)
{
  auto dx = this->out * dout;
  auto sumdx = sum(dx, {1});
  return dx - this->out * sumdx;
}

class SoftmaxWithLoss
{
public:
  SoftmaxWithLoss() {}
  xarray<double> forward(xarray<double> &x, xarray<int> &&t);
  xarray<double> backward(xarray<int> &dout);

private:
  xarray<double> y; // softmaxの出力
  xarray<int> t;    // 教師ラベル
};

inline xarray<double> SoftmaxWithLoss::forward(xarray<double> &x, xarray<int> &&t)
{
  this->t = t;
  this->y = softmax(std::forward<xarray<double>>(x));

  // 教師ラベルがone-hotの場合，正解のインデックスに変換
  if (t.size() == this->y.size())
  {
    t = argmax(xarray<int>{t}, 1);
  }
  xarray<double> loss = cross_entropy_error(std::forward<xarray<double>>(this->y), xarray<int>{t});
  return loss;
}

inline xarray<double> SoftmaxWithLoss::backward(xarray<int> &dout)
{
  int batch_size = this->t.shape()[0];
  int idx_size = this->t.shape()[1];
  xarray<double> dx = this->y;
  // 最適化する必要あり
  // 行列計算ではなく，ただ，1を引くべき場所から引いているだけ
  // index_viewというxtensorのメソッドが使えそう
  for (int i = 0; i < batch_size; i++)
  {
    for (int j = 0; j < idx_size; j++)
    {
      if (t(i, j) == 1)
      {
        dx(i, j) -= 1;
        break;
      }
    }
  }
  dx *= std::forward<xarray<int>>(dout);
  dx = dx / batch_size;
  return dx;
}

class MatMul : public Base
{
public:
  MatMul(xarray<double> W)
  {
    params.push_back(W);
    grads.push_back(zeros<double>(W.shape()));
  }
  virtual xarray<double> forward(xarray<double> &&x);
  virtual xarray<double> backward(xarray<double> &&dout);

private:
  xarray<double> x;
};

inline xarray<double> MatMul::forward(xarray<double> &&x)
{
  xarray<double> W = params[0];
  xarray<double> out = linalg::dot(x, W);
  this->x = x;
  return out;
}

inline xarray<double> MatMul::backward(xarray<double> &&dout)
{
  xarray<double> W = params[0];
  auto dx = linalg::dot(dout, transpose(W));
  xarray<double> dW = linalg::dot(transpose(this->x), dout);
  grads[0] = dW;
  return dx;
}

class Affine : public Base
{
public:
  Affine(xarray<double> W, xarray<double> b)
  {
    params.push_back(W);
    params.push_back(b);
    grads.push_back(zeros<double>(W.shape()));
    grads.push_back(zeros<double>(b.shape()));
  }

  virtual xarray<double> forward(xarray<double> &&x);
  virtual xarray<double> backward(xarray<double> &&dout);

private:
  xarray<double> x;
};

inline xarray<double> Affine::forward(xarray<double> &&x)
{
  xarray<double> W = params[0];
  xarray<double> b = params[1];
  xarray<double> out = linalg::dot(xarray<double>{x}, W) + b;
  this->x = xarray<double>{x};
  return out;
}

inline xarray<double> Affine::backward(xarray<double> &&dout)
{
  xarray<double> W = params[0];
  xarray<double> b = params[1];
  xarray<double> dx = linalg::dot(xarray<double>{dout}, transpose(W));
  xarray<double> dW = linalg::dot(transpose(this->x), xarray<double>{dout});
  xarray<double> db = sum(xarray<double>{dout}, {0});

  this->grads[0] = dW;
  this->grads[1] = db;
  return dx;
}

#endif // LAYERS_H