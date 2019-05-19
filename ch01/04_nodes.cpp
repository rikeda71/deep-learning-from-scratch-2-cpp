#include <iostream>
#include "xtensor/xarray.hpp"
#include "xtensor/xrandom.hpp"
#include "xtensor/xio.hpp"

using namespace std;
using namespace xt;

int main()
{
    const int D = 8, N = 7;

    /*
        RepeatノードとSumノードはお互いが逆の関係にある．
        つまり，Repeatノードの逆伝搬はSumノードの順伝搬，
        Sumノードの逆伝搬はRepeatノードの逆伝搬である
    */

    // Repeat node
    xarray<double> x1 = random::randn<double>({1, D});       // input
    xarray<double> y1 = view(x1, keep(0, 0, 0, 0, 0, 0, 0)); // forward

    xarray<double> dy1 = random::randn<double>({N, D}); // 仮の勾配
    xarray<double> dx1 = xt::sum(dy1, 0);               //  backward

    cout << "Repeat node" << endl;
    cout << x1 << endl;
    cout << y1 << endl;
    cout << dx1 << endl;
    cout << dy1 << endl;

    // Sum node
    xarray<double> x2 = random::randn<double>({1, D}); // input
    xarray<double> y2 = xt::sum(x2, 0);                // forward

    xarray<double> dy2 = random::randn<double>({N, D});        // 仮の勾配
    xarray<double> dx2 = view(dy2, keep(0, 0, 0, 0, 0, 0, 0)); //  backward

    cout << "Sum node" << endl;
    cout << x2 << endl;
    cout << y2 << endl;
    cout << dx2 << endl;
    cout << dy2 << endl;
}