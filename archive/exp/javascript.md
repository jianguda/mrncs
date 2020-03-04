neuralbowmodel
convolutionalmodel
rnnmodel
selfattentionmodel
convselfattentionmodel

###

python train.py --model neuralbow ../resources/data/javascript/final/jsonl/train ../resources/data/javascript/final/jsonl/valid ../resources/data/javascript/final/jsonl/test

```
Test-javascript MRR (bs=1,000):  0.420
FuncNameTest-javascript MRR (bs=1,000):  0.199
Validation-javascript MRR (bs=1,000):  0.415
```

python train.py --model 1dcnn ../resources/data/javascript/final/jsonl/train ../resources/data/javascript/final/jsonl/valid ../resources/data/javascript/final/jsonl/test

```
Test-javascript MRR (bs=1,000):  0.231
FuncNameTest-javascript MRR (bs=1,000):  0.049
Validation-javascript MRR (bs=1,000):  0.232
```

python train.py --model rnn ../resources/data/javascript/final/jsonl/train ../resources/data/javascript/final/jsonl/valid ../resources/data/javascript/final/jsonl/test

```
Test-javascript MRR (bs=1,000):  0.365
FuncNameTest-javascript MRR (bs=1,000):  0.132
Validation-javascript MRR (bs=1,000):  0.364
```

python train.py --model selfatt ../resources/data/javascript/final/jsonl/train ../resources/data/javascript/final/jsonl/valid ../resources/data/javascript/final/jsonl/test

```
Test-javascript MRR (bs=1,000):  0.416
FuncNameTest-javascript MRR (bs=1,000):  0.124
Validation-javascript MRR (bs=1,000):  0.414
```

python train.py --model convselfatt ../resources/data/javascript/final/jsonl/train ../resources/data/javascript/final/jsonl/valid ../resources/data/javascript/final/jsonl/test

```

```
