#include <iostream>
#include "xtensor/xarray.hpp"
#include "xtensor/xio.hpp"
#include "xtensor/xview.hpp"
#include "xtensor-blas/xlinalg.hpp"
#include "../common/util.hpp"

using namespace std;
using namespace xt;

int main() {
    xarray<int> w = {
        {1, 2, 3},
        {4, 5, 6}
    };
    xarray<int> x = {
        {0, 1, 2},
        {3, 4, 5}
    };
    xarray<int> arr1 = w + x;
    xarray<int> arr2 = w * x;
    cout << "足し算" << endl << arr1 << endl;
    cout << "掛け算" << endl << arr2 << endl;

    xarray<int> a = {
        {1, 2},
        {3, 4}
    };
    xarray<int> arr3 = a * 10;
    cout << "broadcast" << endl << arr3 << endl;
    xarray<int> b = {10, 20};
    xarray<int> arr4 = a * b;
    cout << arr4 << endl;

    xarray<int> aa = {1, 2, 3};
    xarray<int> bb = {4, 5, 6};
    xarray<int> arr5 = linalg::dot(aa, bb);
    cout << "ドット積（内積）" << endl << arr5 << endl;

    xarray<int> aaa = {
        {1, 2},
        {3, 4}
    };
    xarray<int> bbb = {
        {5, 6},
        {7, 8}
    };
    xarray<int> arr6 = linalg::dot(aaa, bbb);
    cout << "行列の積" << endl << arr6 << endl;
}