#include <iostream>
#include "xtensor/xarray.hpp"
#include "xtensor/xio.hpp"
#include "../common/util.hpp"

using namespace xt;
using namespace std;


int main() {
    xarray<double> x = {
        1, 2, 3
    };
    cout << x << endl;
    cout << "形状 ";
    cout_vector(x.shape());
    cout << "次元数" << x.dimension() << endl;

    cout << "-------------" << endl;
    xarray<double> w = {
        {1, 2, 3},
        {4, 5, 6}
    };
    cout << w << endl;
    cout << "形状 ";
    cout_vector(w.shape());
    cout << "次元数" << w.dimension() << endl;
    return 0;
}