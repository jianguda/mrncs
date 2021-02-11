`ssh jian@52.142.194.25`
`*************`
scp gpu-test.py jian@52.142.194.25:/home/jian/setup
scp csn-master.zip jian@52.142.194.25:/datadrive
scp jian@52.142.194.25:/datadrive/csn/parser/build/csn-languages.so ./

### run

cd /datadrive/

# sync my code

rm -rf CodeSearchNet/code
cp -r CodeSearchNet/src CodeSearchNet/code

# scp terminal

`scp -r code jian@52.142.194.25:/datadrive/`
`*************`

# ssh terminal

rm -rf CodeSearchNet/code
cp -rf code CodeSearchNet/code

<!-- cp -rf code/. CodeSearchNet/code -->

# delete data

`cd CodeSearchNet/resources/data/`
`find . -name 'contexts.csv' -print | xargs sudo rm`
`find . -name 'counters.pkl' -print | xargs sudo rm`

# clone this repository

`git clone https://github.com/github/CodeSearchNet.git`

cd /datadrive/CodeSearchNet/

# download data (~3.5GB) from S3; build and run the Docker container

`script/setup`

# enter screen

screen

# this will drop you into the shell inside a Docker container

script/console

# optional: log in to W&B to see your training metrics, track your experiments, and submit your models to the benchmark

`wandb login`

# switch the working dir from src to code

pip install nltk tree_sitter scikit-learn

cd ../code/

<!-- python wow.py -->

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
:164

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
