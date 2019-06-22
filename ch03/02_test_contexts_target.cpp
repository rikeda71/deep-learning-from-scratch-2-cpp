#include <iostream>
#include <string>
#include <unordered_map>
#include "xtensor/xarray.hpp"
#include "xtensor/xio.hpp"
#include "../common/util.hpp"

using namespace std;
using namespace xt;

int main()
{
    string text("You say goodbye and I say hello.");
    auto [corpus, w2i, i2w] = preprocess(text);
    unordered_map<string, int>::iterator iter;
    cout << corpus << endl;
    cout_vector(i2w);
    auto [contexts, target] = create_contexts_target(corpus, 1);
    cout << contexts << endl
         << target << endl;
}