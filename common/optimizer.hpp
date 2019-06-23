#ifndef OPTIMIZER_H
#define OPTIMIZER_H

#include <vector>
#include "xtensor/xarray.hpp"
#include "util.hpp"

using namespace std;
using namespace xt;

class OptBase
{
public:
  virtual ~OptBase() {}
  virtual void update(vector<reference_wrapper<xarray<double>>> &params, vector<reference_wrapper<xarray<double>>> grads) = 0;

protected:
  double lr; // learning rate
};

class SGD : public OptBase
{
public:
  SGD(double lr)
  {
    this->lr = lr;
  }
  virtual void update(vector<reference_wrapper<xarray<double>>> &params, vector<reference_wrapper<xarray<double>>> grads);
};

inline void SGD::update(vector<reference_wrapper<xarray<double>>> &params, vector<reference_wrapper<xarray<double>>> grads)
{
  for (int i = 0; i < params.size(); i++)
  {
    params[i].get() -= this->lr * grads[i].get();
  }
}

class Adam : public OptBase
{
  /*
    Adam (http://arxiv.org/abs/1412.6980v8)
   */

private:
  double lr, lr_t;
  double beta1, beta2;
  int iter;
  vector<xt::xarray<double>> m, v;

public:
  Adam(double lr = 0.001, double beta1 = 0.9, double beta2 = 0.999)
  {
    this->lr = lr;
    this->beta1 = beta1;
    this->beta2 = beta2;
    this->iter = 0;
  }

  virtual void update(vector<reference_wrapper<xarray<double>>> &params, vector<reference_wrapper<xarray<double>>> grads);
};

inline void Adam::update(vector<reference_wrapper<xarray<double>>> &params, vector<reference_wrapper<xarray<double>>> grads)
{
  if (this->m.size() == 0)
  {
    for (auto param = params.begin(); param != params.end(); ++param)
    {
      this->m.push_back(xt::zeros_like(param->get()));
      this->v.push_back(xt::zeros_like(param->get()));
    }
  }

  this->iter += 1;
  this->lr_t = this->lr * std::sqrt(1.0 - std::pow(this->beta2, this->iter)) / (1.0 - std::pow(this->beta1, this->iter));

  for (int i = 0; i < params.size(); i++)
  {
    this->m[i] += (1 - this->beta1) * (grads[i].get() - this->m[i]);
    this->v[i] += (1 - this->beta2) * (xt::pow(grads[i].get(), 2) - this->v[i]);

    params[i].get() -= this->lr_t * this->m[i] / (xt::sqrt(this->v[i]) + 1e-7);
  }
}

#endif // OPTIMIZER_HPP