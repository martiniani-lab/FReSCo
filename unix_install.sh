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

alias python=python3
  
git clone https://github.com/flatironinstitute/finufft.git \
&& cd finufft \
&& make lib \
&& make python \
&& cd ..

pip install --no-cache-dir --upgrade pip \
&& pip install --no-cache-dir -r pip_requirements.txt

rm -rf cythonize.dat build

python setup.py build_ext -i -c gcc

#Also add to Python path
#echo "export PYTHONPATH=$PYTHONPATH:/current/directory" >> ~/.bashrc