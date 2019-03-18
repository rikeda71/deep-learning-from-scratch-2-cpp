FROM ubuntu:18.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt update &&\
    apt install -yq \
        wget \
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
    echo "backend: Agg" > ~/.config/matplotlib/matplotlibrc &&\
    wget --quiet https://repo.anaconda.com/archive/Anaconda2-5.3.0-Linux-x86_64.sh -O ~/anaconda.sh &&\
    /bin/bash ~/anaconda.sh -b -p /opt/conda &&\
    rm ~/anaconda.sh &&\
    ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh &&\
    echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc &&\
    echo "conda activate base" >> ~/.bashrc &&\
    . ~/.bashrc &&\
    conda install -c conda-forge xtl
WORKDIR /app

CMD ["/bin/bash"]