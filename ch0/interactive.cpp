#include <iostream>
#include "xtensor/xarray.hpp"
#include "xtensor/xio.hpp"
#include "xtensor/xview.hpp"

int main() {
    xt::xarray<int> arr1 =  {
        {1, 2, 3}
    };
    std::cout << arr1 << std::endl;
    return 0;
}