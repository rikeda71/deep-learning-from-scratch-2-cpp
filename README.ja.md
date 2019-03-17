# deep-learning-from-scratch-2  ~ C++ ver ~
[「ゼロから作る Deep Learning 2」](https://github.com/oreilly-japan/deep-learning-from-scratch-2)のC++バージョンを作成していきます．

このリポジトリを通じて，就職先に必要（そう）なC++と自然言語処理における深層学習の基礎を学習していきます．

## ファイル構造
[公式](https://github.com/oreilly-japan/deep-learning-from-scratch-2)と同様です．

## 使い方

setup
```sh
git clone --recursive https://github.com/s14t284/deep-learning-from-scratch-2-cpp.git
cd deep-learning-from-scratch-2-cpp/
docker build -t dlscratch:latest .
```

compile all files
```sh
make
```

compile a file
```sh
# ex.) ch0/src/main.cpp -> BIN=ch0/src/main
make BIN=/path/to/file compile
```

exec a file
```sh
make BIN=/path/to/file test
```

clean all object file(.o)
```sh
make clean_all
```

clean a object file
```sh
make BIN=/path/to/file clean
```

## LICENSE
MIT

## References
- [deep learning from scratch 2](https://github.com/oreilly-japan/deep-learning-from-scratch-2)
- [arpaka/deep-learning-from-scratch-cpp](https://github.com/arpaka/deep-learning-from-scratch-cpp)