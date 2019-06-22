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

    auto c0 = xt::view(C, w2i["you"], xt::all());
    auto c1 = xt::view(C, w2i["i"], xt::all());
    cout << cos_similarity(c0, c1) << endl;
}