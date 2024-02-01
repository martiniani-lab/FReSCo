FROM ubuntu:22.04
SHELL ["/bin/bash", "-c"]

RUN DEBIAN_FRONTEND=noninteractive \
  apt-get update \
  && apt-get install -y python3 \
  cmake \
  make \
  gcc \
  gfortran \ 
  build-essential \
  libfftw3-dev \
  python3-pip \
  git \
  curl \
  python-is-python3 \
  && rm -rf /var/lib/apt/lists/*
  
RUN alias python=python3
  
WORKDIR /usr/src/app
  
RUN git clone https://github.com/flatironinstitute/finufft.git \
  && cd finufft \
  && make lib \
  && make python \
  && cd ..
  
COPY pip_requirements.txt ./
RUN pip install --no-cache-dir --upgrade pip \
  && pip install --no-cache-dir -r pip_requirements.txt

COPY . .
RUN rm -rf cythonize.dat build
RUN python setup.py build_ext -i -c gcc

RUN echo "export PYTHONPATH=$PYTHONPATH:/usr/src/app" >> ~/.bashrc