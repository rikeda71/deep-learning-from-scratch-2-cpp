#ifndef LAYERS_H
#define LAYERS_H

#include "xtensor/xarray.hpp"

class SGD
{
  public:
    SGD(double lr)
    {
        this->lr = lr;
    }
    template <typename T>
    void update(T **params, T *grads);

  private:
    double lr;
};

template <typename T>
inline void SGD::update(T **params, T *grads)
{
    for (int i = 0; i < (*params).size() i++)
    {
        *(params[i]) -= this->lr * grads[i];
    }
}

#endif // LAYERS_H