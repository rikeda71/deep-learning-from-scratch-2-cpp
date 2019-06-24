COMPILER = g++-7

COMFLAGS = -std=c++17

INCLUDES = \
-I common/ \
-I dataset/ \
-I /usr/include/python3.6/ \
-I /opt/conda/include/python3.7m/ \
-I /opt/conda/lib/python3.7/site-packages/ \
-I /opt/conda/pkgs/xtl-0.6.4-hc9558a2_0/include/ \
-I /opt/conda/pkgs/tk-8.6.8-hbc83047_0/include/ \
-I xtensor/include/ \
-I xtensor-blas/include \
-I xtensor-blas/include/xtensor-blas/flens \
-I matplotlib-cpp/

LIBS = \
-I/opt/conda/include/python3.7m -I/opt/conda/include/python3.7m  -Wno-unused-result -Wsign-compare -march=nocona -mtune=haswell -ftree-vectorize -fPIC -fstack-protector-strong -fno-plt -O3 -ffunction-sections -pipe  -fdebug-prefix-map=/tmp/build/80754af9/python_1553721932202/work=/usr/local/src/conda/python-3.7.3 -fdebug-prefix-map=/opt/conda=/usr/local/src/conda-prefix -fuse-linker-plugin -ffat-lto-objects -flto-partition=none -flto -DNDEBUG -fwrapv -O3 -Wall \
-L/opt/conda/lib/python3.7/config-3.7m-x86_64-linux-gnu -L/opt/conda/lib -lpython3.7m -lcrypt -lpthread -ldl  -lutil -lrt -lm  -Xlinker -export-dynamic \
-Ldataset -lptb.cpython-37m-x86_64-linux-gnu \
-Wl,-rpath=dataset \
-lcblas \
-lblas \
-llapack \
-fPIC \
-fno-lto


TGTS = `ls ch*/*.cpp | sed -e s/.cpp//`
TGT ?= ch0/main
SRCS += $(TGT).cpp
BIN = $(TGT)
OBJS = $(SRCS:.cpp=.o)

all: compile_all

compile: $(BIN)

compile_all:
	docker run --name dlc -v $(PWD)/:/app -id dlscratch:latest
	@for target in $(TGTS); do \
	make TGT=$${target} compile; \
	done
	docker stop dlc
	docker rm dlc

test: $(BIN)
	docker run -it --rm -v $(PWD)/:/app dlscratch:latest ./$(BIN).o

$(BIN): $(OBJS)
	docker exec -it dlc $(COMPILER) $(COMFLAGS) $(INCLUDES) -o $@ $@.cpp $(LIBS)

.cpp.o: $(SRCS)
	docker exec -it dlc $(COMPILER) $(COMFLAGS) $(INCLUDES) -o $@ $< $(LIBS)

clean:
	rm -f $(BIN) $(OBJS)

clean_all:
	@for target in $(TGTS); do \
	make TGT=$${target} clean; \
	done