neuralbowmodel
convolutionalmodel
rnnmodel
selfattentionmodel
convselfattentionmodel

###

python train.py --model neuralbow ../resources/data/php/final/jsonl/train ../resources/data/php/final/jsonl/valid ../resources/data/php/final/jsonl/test

```
Test-php MRR (bs=1,000):  0.566
FuncNameTest-php MRR (bs=1,000):  0.681
Validation-php MRR (bs=1,000):  0.576
```

python train.py --model 1dcnn ../resources/data/php/final/jsonl/train ../resources/data/php/final/jsonl/valid ../resources/data/php/final/jsonl/test

```
Test-php MRR (bs=1,000):  0.537
FuncNameTest-php MRR (bs=1,000):  0.842
Validation-php MRR (bs=1,000):  0.537
```

python train.py --model rnn ../resources/data/php/final/jsonl/train ../resources/data/php/final/jsonl/valid ../resources/data/php/final/jsonl/test

```
Test-php MRR (bs=1,000):  0.600
FuncNameTest-php MRR (bs=1,000):  0.864
Validation-php MRR (bs=1,000):  0.600
```

python train.py --model selfatt ../resources/data/php/final/jsonl/train ../resources/data/php/final/jsonl/valid ../resources/data/php/final/jsonl/test

```
Test-php MRR (bs=1,000):  0.588
FuncNameTest-php MRR (bs=1,000):  0.831
Validation-php MRR (bs=1,000):  0.597
```

python train.py --model convselfatt ../resources/data/php/final/jsonl/train ../resources/data/php/final/jsonl/valid ../resources/data/php/final/jsonl/test

```

```
