neuralbowmodel
convolutionalmodel
rnnmodel
selfattentionmodel
convselfattentionmodel

###

python train.py --model neuralbow ../resources/data/java/final/jsonl/train ../resources/data/java/final/jsonl/valid ../resources/data/java/final/jsonl/test

```
Test-java MRR (bs=1,000):  0.583
FuncNameTest-java MRR (bs=1,000):  0.673
Validation-java MRR (bs=1,000):  0.572
```

python train.py --model 1dcnn ../resources/data/java/final/jsonl/train ../resources/data/java/final/jsonl/valid ../resources/data/java/final/jsonl/test

```
Test-java MRR (bs=1,000):  0.522
FuncNameTest-java MRR (bs=1,000):  0.753
Validation-java MRR (bs=1,000):  0.498
```

python train.py --model rnn ../resources/data/java/final/jsonl/train ../resources/data/java/final/jsonl/valid ../resources/data/java/final/jsonl/test

```
Test-java MRR (bs=1,000):  0.586
FuncNameTest-java MRR (bs=1,000):  0.801
Validation-java MRR (bs=1,000):  0.562
```

python train.py --model selfatt ../resources/data/java/final/jsonl/train ../resources/data/java/final/jsonl/valid ../resources/data/java/final/jsonl/test

```
Test-java MRR (bs=1,000):  0.555
FuncNameTest-java MRR (bs=1,000):  0.752
Validation-java MRR (bs=1,000):  0.531
```

python train.py --model convselfatt ../resources/data/java/final/jsonl/train ../resources/data/java/final/jsonl/valid ../resources/data/java/final/jsonl/test

```

```
