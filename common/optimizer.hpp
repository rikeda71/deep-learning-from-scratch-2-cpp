#ifndef OPTIMIZER_H
#define OPTIMIZER_H

#include <vector>
#include "xtensor/xarray.hpp"
#include "util.hpp"

using namespace std;

class SGD
{
public:
  SGD(double lr)
  {
    this->lr = lr;
  }
  template <typename T>
  void update(vector<vector<T>> &params, vector<vector<T>> grads);
  template <typename T>
  void update(vector<reference_wrapper<vector<T>>> &params, vector<reference_wrapper<vector<T>>> grads);

private:
  double lr;
};

template <typename T>
inline void SGD::update(vector<vector<T>> &params, vector<vector<T>> grads)
{
  for (int i = 0; i < params.size(); i++)
  {
    for (int j = 0; j < params[i].size(); j++)
    {
      params[i][j] -= this->lr * grads[i][j];
    }
  }
}

template <typename T>
inline void SGD::update(vector<reference_wrapper<vector<T>>> &params, vector<reference_wrapper<vector<T>>> grads)
{
  for (int i = 0; i < params.size(); i++)
  {
    for (int j = 0; j < params[i].get().size(); j++)
    {
      params[i].get()[j] -= this->lr * grads[i].get()[j];
    }
  }
}

#endif // OPTIMIZER_HPP