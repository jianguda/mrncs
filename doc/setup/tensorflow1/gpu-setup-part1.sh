#!/bin/bash

sudo add-apt-repository -y ppa:deadsnakes/ppa
sudo apt-get -y update
sudo apt-get -y install python3.6
sudo apt-get -y install python-pip
sudo pip install --upgrade virtualenv
sudo apt-get -y install openjdk-8-jdk git python-dev python3-dev python-numpy python3-numpy build-essential python-pip python3-pip python-virtualenv swig python-wheel libcurl3-dev curl
sudo apt-get -y install ffmpeg
curl -O http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/cuda-repo-ubuntu1604_9.0.176-1_amd64.deb
sudo apt-key adv --fetch-keys http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/7fa2af80.pub
sudo dpkg -i ./cuda-repo-ubuntu1604_9.0.176-1_amd64.deb
sudo apt-get -y update
sudo apt-get -y install cuda-9-0
echo >> ~/.bashrc '
alias python=python3.6
'
source ~/.bashrc
sudo reboot
