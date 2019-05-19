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
  virtual void update(vector<xarray<double>> &params, vector<xarray<double>> grads) = 0;
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
  virtual void update(vector<xarray<double>> &params, vector<xarray<double>> grads);
  virtual void update(vector<reference_wrapper<xarray<double>>> &params, vector<reference_wrapper<xarray<double>>> grads);
};

inline void SGD::update(vector<xarray<double>> &params, vector<xarray<double>> grads)
{
  for (int i = 0; i < params.size(); i++)
  {
    params[i] -= this->lr * grads[i];
  }
}

inline void SGD::update(vector<reference_wrapper<xarray<double>>> &params, vector<reference_wrapper<xarray<double>>> grads)
{
  for (int i = 0; i < params.size(); i++)
  {
    params[i].get() -= this->lr * grads[i].get();
  }
}

#endif // OPTIMIZER_HPP