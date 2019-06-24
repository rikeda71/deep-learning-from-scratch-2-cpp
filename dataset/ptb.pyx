# cython: language_level=3, distutils: language = c++
import pickle
import sys
import os
sys.path.append('..')
import urllib.request
from libcpp.unordered_map cimport unordered_map
from libcpp.vector cimport vector
from libcpp.string cimport string

url_base = 'https://raw.githubusercontent.com/tomsercu/lstm/master/data/'
key_file = {
    'train':'ptb.train.txt',
    'test':'ptb.test.txt',
    'valid':'ptb.valid.txt'
}
save_file = {
    'train':'ptb.train.npy',
    'test':'ptb.test.npy',
    'valid':'ptb.valid.npy'
}
vocab_file = 'ptb.vocab.pkl'
dataset_dir = os.path.dirname(os.path.abspath(__file__))

cpdef public _download(const char* file_name):
    file_path = dataset_dir + '/' + file_name

    if os.path.exists(file_path):
        return

    print('Downloading ' + file_name + ' ... ')
    urllib.request.urlretrieve(url_base + file_name, file_path)
    print('Done')

cdef public bint file_exists(const char* path):
    if os.path.exists(path):
        return True
    return False

cdef public vector[string] return_dataset(const char* file_path):
    cdef vector[string] words
    word_list = open(file_path).read().replace('\n', '<eos>').strip().split()
    for word in word_list:
        words.push_back(word)
    return words

cdef public struct Word_id_pair:
    unordered_map[string, int] word_to_id
    vector[string] id_to_word

# cdef public (unordered_map[string, int], vector[string]) load_vocab():
cdef public Word_id_pair load_vocab():
    vocab_path = dataset_dir + '/' + vocab_file
    data_type = 'train'
    file_name = key_file[data_type]
    file_path = dataset_dir + '/' + file_name
    cdef unordered_map[string, int] word_to_id
    cdef vector[string] id_to_word
    cdef list words = [1]
    cdef Word_id_pair wip

    cout << "valiables" << endl

    if os.path.exists(vocab_path):
        with open(vocab_path, 'rb') as f:
            word_to_id, id_to_word = pickle.load(f)
        return word_to_id, id_to_word

    cout << "download" << endl
    _download(file_name)

    words = open(file_path).read().replace('\n', '<eos>').strip().split()

    for i, word in enumerate(words):
        if word_to_id[word].count == 0:
            tmp_id = len(word_to_id)
            word_to_id[word] = tmp_id
            id_to_word.push_back(word)

    with open(vocab_path, 'wb') as f:
        pickle.dump((word_to_id, id_to_word), f)

    wip.word_to_id = word_to_id
    wip.id_to_word = id_to_word
    return wip
