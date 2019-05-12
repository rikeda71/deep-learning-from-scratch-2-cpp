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
  void update(vector<vector<T>> params, vector<vector<T>> grads);

  template <typename T>
  void update(T &params, T &&grads);

private:
  double lr;
};

template <typename T>
inline void SGD::update(vector<vector<T>> params, vector<vector<T>> grads)
{
  for (int i = 0; i < params.size(); i++)
  {
    for (int j = 0; j < params[i].size(); j++)
    {
      SGD::update(params[i][j], T{grads[i][j]});
    }
  }
}

template <typename T>
inline void SGD::update(T &params, T &&grads)
{
  // cout << T{params} << endl;
  // cout << T{grads} << endl;
  // TODO: ここのgradsの値が全て0になっている．どこかで値の代入がリセットされている
  cout << this->lr * std::forward<T>(grads) << endl;
  params -= this->lr * std::forward<T>(grads);
}

#endif // OPTIMIZER_HPP