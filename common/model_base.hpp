#ifndef MODEL_BASE_HPP
#define MODEL_BASE_HPP

#include <vector>
#include "xtensor/xarray.hpp"

using namespace std;
using namespace xt;

class ModelBase
{
public:
    virtual ~ModelBase() {}
    virtual xarray<double> predict(xarray<double> &&x) = 0;
    virtual xarray<double> forward(xarray<double> &x, xarray<int> &t) = 0;
    virtual xarray<double> backward(xarray<int> &&dout) = 0;
    vector<reference_wrapper<xarray<double>>> params, grads;

protected:
    vector<reference_wrapper<Base>> layers;
};

#endif