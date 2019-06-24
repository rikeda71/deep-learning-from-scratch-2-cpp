#include <iostream>
#include <fstream>
#include <string>
#include <tuple>
#include <map>
#include "xtensor/xarray.hpp"
#include "xtensor/xnpy.hpp"
#include "Python.h"
#include "../dataset/ptb.h"

using namespace std;

namespace ptb
{
map<string, string> SAVE_FILE{
    {"train", "ptb.train.npy"},
    {"test", "ptb.test.npy"},
    {"valid", "ptb.valid.npy"}};

map<string, string> KEY_FILE{
    {"train", "ptb.train.txt"},
    {"test", "ptb.test.txt"},
    {"valid", "ptb.valid.txt"}};

tuple<xt::xarray<double>, unordered_map<string, int>, vector<string>> load_data(string data_type = "train", string dataset_dir = "./")
{

    if (data_type == "val")
    {
        data_type = "valid";
    }
    string save_path = dataset_dir + SAVE_FILE[data_type];

    Py_Initialize();
    PyInit_libptb();
    struct Word_id_pair wip = load_vocab();
    auto word_to_id = wip.word_to_id;
    auto id_to_word = wip.id_to_word;
    cout << "word_to_id_to_word" << endl;
    // auto [word_to_id, id_to_word] = load_vocab();
    if (file_exists(save_path.c_str()))
    {
        xt::xarray<double> corpus = xt::load_npy<double>(save_path);
        return {corpus, word_to_id, id_to_word};
    }
    string file_name = KEY_FILE[data_type];
    string file_path = dataset_dir + file_name;
    _download(file_name.c_str(), 0);
    vector<string> words = return_dataset(file_path.c_str());
    Py_Finalize();
    vector<int> ids;
    for (int i = 0; i < words.size(); i++)
    {
        ids.push_back(word_to_id[words[i]]);
    }
    xt::xarray<double> corpus = xt::adapt(ids);
    xt::dump_npy(save_path, corpus);
    return {corpus, word_to_id, id_to_word};
}

} // namespace ptb