neuralbowmodel
convolutionalmodel
rnnmodel
selfattentionmodel
convselfattentionmodel

### 

python train.py --model neuralbow ../resources/data/python/final/jsonl/train ../resources/data/python/final/jsonl/valid ../resources/data/python/final/jsonl/test

```
Test-python MRR (bs=1,000):  0.642
FuncNameTest-python MRR (bs=1,000):  0.524
Validation-python MRR (bs=1,000):  0.608
```

python train.py --model 1dcnn ../resources/data/python/final/jsonl/train ../resources/data/python/final/jsonl/valid ../resources/data/python/final/jsonl/test

```
Test-python MRR (bs=1,000):  0.515
FuncNameTest-python MRR (bs=1,000):  0.547
Validation-python MRR (bs=1,000):  0.476
```

python train.py --model rnn ../resources/data/python/final/jsonl/train ../resources/data/python/final/jsonl/valid ../resources/data/python/final/jsonl/test

```
Test-python MRR (bs=1,000):  0.635
FuncNameTest-python MRR (bs=1,000):  0.635
Validation-python MRR (bs=1,000):  0.585
```

python train.py --model selfatt ../resources/data/python/final/jsonl/train ../resources/data/python/final/jsonl/valid ../resources/data/python/final/jsonl/test

```
Test-python MRR (bs=1,000):  0.665
FuncNameTest-python MRR (bs=1,000):  0.630
Validation-python MRR (bs=1,000):  0.621
```

python train.py --model convselfatt ../resources/data/python/final/jsonl/train ../resources/data/python/final/jsonl/valid ../resources/data/python/final/jsonl/test

```
```
