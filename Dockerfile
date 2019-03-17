FROM ubuntu:18.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt update &&\
    apt install -yq \
        g++ \
        cmake \
        make \
        gdb \
        python-matplotlib \
        python-numpy \
        python-dev \
        python-tk \
        build-essential \
        libgtest-dev \
        libopenblas-dev \
        liblapack-dev \
        xtensor-dev &&\
    apt clean &&\
    rm -rf /var/lib/apt/lists/* &&\
    mkdir /app &&\
    mkdir -p ~/.config/matplotlib/ &&\
    echo "backend: Agg" > ~/.config/matplotlib/matplotlibrc
WORKDIR /app

CMD ["/bin/bash"]