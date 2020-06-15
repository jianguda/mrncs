`ssh jian@40.112.67.232`
`************`
scp gpu-test.py jian@52.142.194.25:/home/jian/setup
scp csn-master.zip jian@52.142.194.25:/datadrive
scp jian@52.142.194.25:/datadrive/csn/parser/build/csn-languages.so ./

### run

<!-- cd /datadrive/ -->

# sync my code

sudo rm -rf /datadrive/core

<!-- cp -r CodeSearchNet/src CodeSearchNet/code -->

# scp terminal

`scp -r core jian@40.112.67.232:/datadrive/`
`************`

# ssh terminal

rm -rf /datadrive/codesnippetsearch/rok
cp -rf /datadrive/core/rok /datadrive/codesnippetsearch/rok

<!-- cp -rf code/. CodeSearchNet/code -->

# delete data

`cd CodeSearchNet/resources/data/`
`find . -name 'contexts.csv' -print | xargs sudo rm`
`find . -name 'counters.pkl' -print | xargs sudo rm`

# clone this repository

`git clone https://github.com/novoselrok/codesnippetsearch.git`

cd /datadrive/codesnippetsearch/

# download data (~3.5GB) from S3; build and run the Docker container

`script/setup`

# enter screen

screen

# virtual environment

# https://packaging.python.org/guides/installing-using-pip-and-virtual-environments/

<!-- pip3 install --user virtualenv -->
<!-- sudo apt-get install python3-venv -->
<!-- python3 -m venv env -->

source env/bin/activate

<!-- pip3 install --upgrade pip -->
<!-- pip3 install --upgrade setuptools -->
<!-- pip3 install -r requirements.txt -->
<!-- which python -->
<!-- deactivate -->

<!-- touch env/lib/python3.6/site-packages/.pth
/datadrive/codesnippetsearch/ -->

# follow steps in README

<!-- wget https://raw.githubusercontent.com/github/CodeSearchNet/master/resources/queries.csv -->

<!-- cd /datadrive/codesnippetsearch/
gzip -kdr resources/data -->

<!-- cd /datadrive/codesnippetsearch/code_search -->

<!-- cd /datadrive/codesnippetsearch/rok
python3 prepare_data.py --prepare-all -->

# train model

cd rok

<!-- cd code_search -->

python3 run.py fully
python3 run.py train evaluate

# switch between siamese and raw

vi shared.py
:4

<!-- python train.py --model treepath ../resources/saved_models ../resources/data/python/final/jsonl/train ../resources/data/python/final/jsonl/valid ../resources/data/python/final/jsonl/test -->

python3 evaluate_model.py -r jianguda/CodeSearchNet/01234567

# for different languages

vi predict.py
:116
:134

# for attention

vi encoders/tree/tree_seq_encoder.py
:205

# for preprocessing

vi encoders/tree/common.py
:112
:132
vi encoders/seq_encoder.py
:37

# for distance metric

vi models/tree_raw_model.py
:21
vi encoders/tree/tree_seq_encoder.py
:204

# for multi-modal

vi scripts/ts.py
:63

<!-- python train.py --model treeraw ../resources/saved_models ../resources/data/python/final/jsonl/train ../resources/data/python/final/jsonl/valid ../resources/data/python/final/jsonl/test -->

<!-- python train.py --model treeleaf ../resources/saved_models ../resources/data/python/final/jsonl/train ../resources/data/python/final/jsonl/valid ../resources/data/python/final/jsonl/test -->

python train.py --model treepath ../resources/saved_models ../resources/data/python/final/jsonl/train ../resources/data/python/final/jsonl/valid ../resources/data/python/final/jsonl/test

python predict.py -r jianguda/CodeSearchNet/01234567

<!-- python train.py --model treeraw ../resources/saved_models -->

# verify your setup by training a tiny model

python train.py --testrun

# see other command line options, try a full training run with default values,

# and explore other model variants by extending this baseline script

python train.py --help
python train.py

python train.py --model neuralbow
python train.py --model 1dcnn
python train.py --model rnn
python train.py --model selfatt
python train.py --model convselfatt

# generate predictions for model evaluation

python predict.py -r jianguda/CodeSearchNet/0123456 # this is the org/project_name/run_id
