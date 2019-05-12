COMPILER = g++-7

COMFLAGS = -std=c++17

INCLUDES = \
-I common/ \
-I dataset/ \
-I /usr/include/python3.6/ \
-I /opt/conda/pkgs/xtl-0.6.2-hc9558a2_0/include/ \
-I xtensor/include/ \
-I xtensor-blas/include \
-I xtensor-blas/include/xtensor-blas/flens \
-I matplotlib-cpp/

LIBS = \
-lpython3.6m \
-lcblas \
-lblas


TGTS = `ls ch*/*.cpp | sed -e s/.cpp//`
TGT ?= ch0/main
SRCS += $(TGT).cpp
BIN = $(TGT)
OBJS = $(SRCS:.cpp=.o)

all: compile_all

compile: $(BIN)

compile_all:
	@for target in $(TGTS); do \
	make TGT=$${target} compile; \
	done

test: $(BIN)
	docker run -it --rm -v $(PWD)/:/app dlscratch:latest ./$(BIN).o

$(BIN): $(OBJS)
	docker run -it --rm -v $(PWD)/:/app dlscratch:latest $(COMPILER) $(COMFLAGS) $(INCLUDES) -o $@ $@.cpp $(LIBS)

.cpp.o: $(SRCS)
	docker run -it --rm -v $(PWD)/:/app dlscratch:latest $(COMPILER) $(COMFLAGS) $(INCLUDES) -o $@ $< $(LIBS)

clean:
	rm -f $(BIN) $(OBJS)

clean_all:
	@for target in $(TGTS); do \
	make TGT=$${target} clean; \
	done