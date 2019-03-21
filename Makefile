COMPILER = g++-7

COMFLAGS = -std=c++14

INCLUDES = \
-I common/ \
-I /usr/include/python2.7/ \
-I /opt/conda/pkgs/xtl-0.6.0-h6bb024c_0/include/ \
-I xtensor/include/ \
-I xtensor-blas/include \
-I xtensor-blas/include/xtensor-blas/flens \
-I matplotlib-cpp/

LIBS = \
-lpython2.7 \
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