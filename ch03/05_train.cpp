#include <iostream>
#include <string>
#include <unordered_map>
#include "xtensor/xarray.hpp"
#include "xtensor/xio.hpp"
#include "../common/util.hpp"
#include "../common/optimizer.hpp"
#include "../common/trainer.hpp"
#include "04_simple_cbow.hpp"
#include "06_simple_skip_gram.hpp"

using namespace std;
using namespace xt;

int main()
{
    const int window_size = 1;
    const int hidden_size = 5;
    const int batch_size = 3;
    const int max_epoch = 1000;

    string text("You say goodbye and I say hello.");
    auto [corpus, w2i, i2w] = preprocess(text);

    int vocab_size = w2i.size();
    auto [contexts, target] = create_contexts_target(corpus, window_size);
    contexts = convert_one_hot(contexts, vocab_size);
    target = convert_one_hot(target, vocab_size);

    auto model = SimpleCBOW(vocab_size, hidden_size);
    // auto model = SimpleSkipGram(vocab_size, hidden_size);
    auto optimizer = Adam();
    auto trainer = Trainer(&model, &optimizer);

    trainer.fit(contexts, target, max_epoch, batch_size);
    trainer.plot();

    string word;
    auto word_vecs = *(model.word_vecs);
    for (int i = 0; i < i2w.size(); i++)
    {
        word = i2w[i];
        cout << word << xt::view(word_vecs, i, xt::all()) << endl;
    }
}