#include <tuple>
#include "xtensor/xarray.hpp"
#include "matplotlibcpp.h"
#include "../dataset/spiral.hpp"

namespace plt = matplotlibcpp;

int main()
{
    auto [x, t] = load_data();
    vector<double> points_x1;
    vector<double> points_y1;
    vector<double> points_x2;
    vector<double> points_y2;
    vector<double> points_x3;
    vector<double> points_y3;

    // データ点のプロット
    const int N = 100;
    const int CLS_NUM = 3;
    const char markers[3] = {'o', 'x', '^'};
    int i = 0;
    for (i = 0; i < N; i++)
    {
        points_x1.push_back(x[0][i]);
        points_y1.push_back(x[1][i]);
    }
    for (i = N; i < N * 2; i++)
    {
        points_x2.push_back(x[0][i]);
        points_y2.push_back(x[1][i]);
    }
    for (i = N * 2; i < N * 3; i++)
    {
        points_x3.push_back(x[0][i]);
        points_y3.push_back(x[1][i]);
    }

    plt::scatter(points_x1, points_y1, 40);
    plt::scatter(points_x2, points_y2, 40);
    plt::scatter(points_x3, points_y3, 40);

    plt::save("./ch1/spiral_dataset.png");
    return 0;
}