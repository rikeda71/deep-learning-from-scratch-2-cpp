#ifndef LAYERS_H
#define LAYERS_H

#include <vector>
#include "xtensor/xarray.hpp"
#include "xtensor/xrandom.hpp"
#include "xtensor/xsort.hpp"
#include "xtensor-blas/xlinalg.hpp"
#include "./functions.hpp"

using namespace std;
using namespace xt;

// 基底クラス
class Base
{
public:
  vector<xarray<double>> params, grads;
  // 基底クラスを定義するときは`~Class名() {}`を定義する必要がある
  virtual ~Base() {}
  template <typename T>
  virtual T forward(T x) = 0;
  virtual T backward(T dout) = 0;
};

class Sigmoid : public Base
{
public:
  Sigmoid() {}
  template <typename T>
  virtual T forward(T x);
  virtual T backward(T dout);

private:
  xarray<double> out;
};

template <typename T>
inline T Sigmoid::forward(T x)
{
  x = 1 / (1 + exp(-x));
  out = x;
  return x;
}

template <typename T>
inline T Sigmoid::backward(T dout)
{
  auto dx = dout * (1.0 - out) * out;
  return dx;
}

class Softmax : public Base
{
public:
  Softmax() {}
  template <typename T>
  virtual T forward(T x);
  virtual T backward(T dout);

private:
  xarray<double> out;
};

template <typename T>
inline T Softmax::forward(T x)
{
  out = softmax(x);
  return softmax(x);
}

template <typename T>
inline T Softmax::backward(T dout)
{
  auto dx = out * dout;
  auto sumdx = sum(dx, {1});
  dx = dx - out * sumdx;
  return dx;
}

class SoftmaxWithLoss : public Base
{
public:
  SoftmaxWithLoss() {}
  template <typename T>
  virtual T forward(T x);
  virtual T backward(T dout);

private:
  xarray<double> y; // softmaxの出力
  xarray<int> t;    // 教師ラベル
};

template <typename T>
inline T SoftmaxWithLoss::forward(T x)
{
  t = t;
  y = softmax(x);

  // 教師ラベルがone-hotの場合，正解のインデックスに変換
  if (t.size() == y.size())
  {
    t = argmax(t, 1);
  }
  auto loss = cross_entropy_error(y, t);
  return loss;
}

template <typename T>
inline T SoftmaxWithLoss::backward(T dout)
{
  int batch_size = t.shape()[0];

  auto dx = y;
  dx[arange(batch_size), t] = dx[arange(batch_size), t] - 1;
  dx = dx * dout;
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
  template <typename T>
  virtual T forward(T x);
  virtual T backward(T dout);

private:
  xarray<double> x;
};

template <typename T>
inline T MatMul::forward(T x)
{
  xarray<double> W = params[0];
  xarray<double> out = linalg::dot(x, W);
  this->x = x;
  return out;
}

template <typename T>
inline T MatMul::backward(T dout)
{
  xarray<double> W = params[0];
  auto dx = linalg::dot(dout, transpose(W));
  xarray<double> dW = linalg::dot(transpose(x), dout);
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
  template <typename T>
  virtual T forward(T x);
  virtual T backward(T dout);

private:
  xarray<double> x;
};

template <typename T>
inline T Affine::forward(T x)
{
  xarray<double> W = params[0];
  xarray<double> b = params[1];
  auto out = linalg::dot(x, W) + b;
  this->x = x;
  return out;
}

template <typename T>
inline T Affine::backward(T dout)
{
  xarray<double> W = params[0];
  xarray<double> b = params[1];
  auto dx = linalg::dot(dout, transpose(W));
  auto dW = linalg::dot(transpose(x), dout);
  auto db = sum(dout, {0});

  grads.clear();
  grads.shrink_to_fit();
  grads.push_back(dW);
  grads.push_back(db);
  return dx;
}

#endif // LAYERS_H