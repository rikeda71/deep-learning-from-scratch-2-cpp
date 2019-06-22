#include <iostream>
#include <string>
#include <unordered_map>
#include "xtensor/xarray.hpp"
#include "xtensor/xio.hpp"
#include "../common/util.hpp"

using namespace std;
using namespace xt;


int main() {
    string text("You say goodbye and I say hello.");
    auto [corpus, w2i, i2w] = preprocess(text);
    unordered_map<string, int>::iterator iter;
    cout << corpus << endl;
    for (iter = w2i.begin(); iter != w2i.end(); iter++) {
        cout << iter->first << " " << iter->second << endl;
    }
    cout_vector(i2w);
}