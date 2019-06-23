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
    auto [contexts, target] = create_contexts_target(corpus, 1);
    int vocab_size = w2i.size();
    cout << "convert contexts" << endl;
    contexts = convert_one_hot(contexts, vocab_size);
    cout << "convert target" << endl;
    target = convert_one_hot(target, vocab_size);
    cout << contexts << endl
         << target << endl;
}