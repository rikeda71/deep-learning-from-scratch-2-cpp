#include <iostream>
#include <string>
#include <vector>
#include "xtensor/xarray.hpp"
#include "xtensor/xio.hpp"
#include "../common/util.hpp"

using namespace std;

int main()
{
    string text = "You say goodbye and I say hello.";
    auto [corpus, w2i, i2w] = preprocess(text);
    int vocab_size = w2i.size();
    auto C = create_co_matrix(corpus, vocab_size);
    auto W = ppmi(C);

    cout << "coveriance matrix" << endl
         << C << endl
         << string(50, '-') << endl
         << "PPMI" << endl
         << std::setprecision(3) << W << endl;
}