# How to Reproduce Experiments

## Dowload Data

```bash
mkdir -p resources/data
cd resources/data
wget https://s3.amazonaws.com/code-search-net/CodeSearchNet/v2/{python,java,go,php,javascript,ruby}.zip
wget https://raw.githubusercontent.com/github/CodeSearchNet/master/resources/queries.csv
```

## Setup Environment

to satisfy the following settings, or compatient alternatives

```markdown
CUDA@10.0
CUDNN@7.4.2
Python@3.6
TensorFlow@2.1
```

```bash
pip3 install -r requirements.txt
```

## Execute Programs

```bash
cd code/tree
python3 run.py fully
```

to config the programs to run, check the parameters in `shared.py`

Because of an unknown issue in my implementation, I usually run following commands to get results. The results produced by the first run is incorrect, so the second run is required

```bash
python3 run.py train evaluate
python3 run.py evaluate
```

## Check Results

training datails and results will be uploaded to WANDB
