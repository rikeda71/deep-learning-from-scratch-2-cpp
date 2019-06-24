FROM ubuntu:18.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt update &&\
    apt install -yq \
    wget \
    g++ \
    cmake \
    make \
    gdb \
    python3-numpy \
    python3-dev \
    build-essential \
    libgtest-dev \
    libopenblas-dev \
    libatlas-base-dev \
    liblapack-dev &&\
    apt clean &&\
    rm -rf /var/lib/apt/lists/* &&\
    mkdir /app &&\
    mkdir -p ~/.config/matplotlib/ &&\
    echo "backend: Agg" > ~/.config/matplotlib/matplotlibrc &&\
    wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh &&\
    /bin/bash ~/miniconda.sh -b -p /opt/conda &&\
    rm ~/miniconda.sh &&\
    ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh &&\
    echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc &&\
    . ~/.bashrc &&\
    conda install -c conda-forge xtl &&\
    conda install -c conda-forge cython &&\
    conda install -c conda-forge tk &&\
    conda install -c conda-forge matplotlib
WORKDIR /app



CMD ["/bin/bash"]