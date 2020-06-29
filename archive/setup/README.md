# TensorFlow1.12@Ubuntu16.04

based on https://github.com/AndyYSWoo/Azure-GPU-Setup

1. follow the pdf file at the `tensorflow1` floder
2. attach a disk following [this link](https://docs.microsoft.com/zh-cn/previous-versions/azure/virtual-machines/linux/classic/attach-disk-classic)
3. check `nvidia-smi`
4. install docker following [this link](https://linuxize.com/post/how-to-install-and-use-docker-on-ubuntu-18-04/)
5. btw, install nvidia-container-toolkit following [this link](https://github.com/NVIDIA/nvidia-docker#ubuntu-16041804-debian-jessiestretchbuster)
6. quickstart following [this link](https://github.com/jianguda/CodeSearchNet#quickstart)
7. sudo apt install python3.6-dev
   python -m pip install --upgrade tree-sitter

Cuda@9.0
CuDNN@7.4.2
Python@3.6.10
TensorFlow@1.12

# TensorFlow2.1@Ubuntu18.04

based on https://github.com/lmtoan/azure-vm-setup

1. follow the README file at the `tensorflow2` floder, only need to read the "Instructions" section.
2. attach a disk following [this link](https://docs.microsoft.com/zh-cn/previous-versions/azure/virtual-machines/linux/classic/attach-disk-classic)
3. check `nvidia-smi`
4. quickstart following [this link](https://github.com/novoselrok/codesnippetsearch)

CUDA@10.0
CUDNN@7.4.1.5
Python@3.6.9
TensorFlow@2.1
