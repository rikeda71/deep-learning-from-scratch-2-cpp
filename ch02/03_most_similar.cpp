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

    most_similar("you", w2i, i2w, C, 5);
}