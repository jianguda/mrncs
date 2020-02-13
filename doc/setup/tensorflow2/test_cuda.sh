curl -O https://github.com/AndyYSWoo/Azure-GPU-Setup/raw/master/libcudnn7-doc_7.0.5.15-1%2Bcuda9.0_amd64.deb

sudo dpkg -i libcudnn7-doc_7.0.5.15-1+cuda9.0_amd64.deb

cp -r /usr/src/cudnn_samples_v7/ .
cd cudnn_samples_v7/mnistCUDNN
make clean && make
./mnistCUDNN
