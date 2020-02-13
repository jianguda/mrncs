#!/bin/bash

curl -O https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1604/x86_64/libcudnn7-dev_7.4.2.24-1+cuda9.0_amd64.deb
curl -O https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1604/x86_64/libcudnn7_7.4.2.24-1+cuda9.0_amd64.deb

sudo dpkg -i libcudnn7_7.4.2.24-1+cuda9.0_amd64.deb
sudo dpkg -i libcudnn7-dev_7.4.2.24-1+cuda9.0_amd64.deb
sudo dpkg -i libcudnn7-doc_7.4.2.24-1+cuda9.0_amd64.deb

echo >> ~/.bashrc '
export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/usr/local/cuda/lib64:/usr/local/cuda/extras/CUPTI/lib64"
export CUDA_HOME=/usr/local/cuda
export PATH="$PATH:/usr/local/cuda/bin"
'
source ~/.bashrc

sudo apt-get -y install cmake
sudo apt-get -y install zlib1g-dev
python3.6 -m pip install --upgrade -r requirements.txt

cd cudnn_samples_v7/mnistCUDNN
make clean && make
./mnistCUDNN
