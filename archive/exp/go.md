neuralbowmodel
convolutionalmodel
rnnmodel
selfattentionmodel
convselfattentionmodel

### 

python train.py --model neuralbow ../resources/data/go/final/jsonl/train ../resources/data/go/final/jsonl/valid ../resources/data/go/final/jsonl/test

```
Test-go MRR (bs=1,000):  0.664
FuncNameTest-go MRR (bs=1,000):  0.302
Validation-go MRR (bs=1,000):  0.802
```

python train.py --model 1dcnn ../resources/data/go/final/jsonl/train ../resources/data/go/final/jsonl/valid ../resources/data/go/final/jsonl/test

```
Test-go MRR (bs=1,000):  0.695
FuncNameTest-go MRR (bs=1,000):  0.089
Validation-go MRR (bs=1,000):  0.785
```

python train.py --model rnn ../resources/data/go/final/jsonl/train ../resources/data/go/final/jsonl/valid ../resources/data/go/final/jsonl/test

```
Test-go MRR (bs=1,000):  0.704
FuncNameTest-go MRR (bs=1,000):  0.115
Validation-go MRR (bs=1,000):  0.787
```

python train.py --model selfatt ../resources/data/go/final/jsonl/train ../resources/data/go/final/jsonl/valid ../resources/data/go/final/jsonl/test

```
Test-go MRR (bs=1,000):  0.729
FuncNameTest-go MRR (bs=1,000):  0.166
Validation-go MRR (bs=1,000):  0.822
```

python train.py --model convselfatt ../resources/data/go/final/jsonl/train ../resources/data/go/final/jsonl/valid ../resources/data/go/final/jsonl/test

```
```
