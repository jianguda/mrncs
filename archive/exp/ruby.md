neuralbowmodel
convolutionalmodel
rnnmodel
selfattentionmodel
convselfattentionmodel

###

python train.py --model neuralbow ../resources/data/ruby/final/jsonl/train ../resources/data/ruby/final/jsonl/valid ../resources/data/ruby/final/jsonl/test

```
Test-ruby MRR (bs=1,000):  0.320
FuncNameTest-ruby MRR (bs=1,000):  0.283
Validation-ruby MRR (bs=1,000):  0.357
```

python train.py --model 1dcnn ../resources/data/ruby/final/jsonl/train ../resources/data/ruby/final/jsonl/valid ../resources/data/ruby/final/jsonl/test

```
Test-ruby MRR (bs=1,000):  0.111
FuncNameTest-ruby MRR (bs=1,000):  0.154
Validation-ruby MRR (bs=1,000):  0.118
```

python train.py --model rnn ../resources/data/ruby/final/jsonl/train ../resources/data/ruby/final/jsonl/valid ../resources/data/ruby/final/jsonl/test

```
Test-ruby MRR (bs=1,000):  0.220
FuncNameTest-ruby MRR (bs=1,000):  0.359
Validation-ruby MRR (bs=1,000):  0.242
```

python train.py --model selfatt ../resources/data/ruby/final/jsonl/train ../resources/data/ruby/final/jsonl/valid ../resources/data/ruby/final/jsonl/test

```
Test-ruby MRR (bs=1,000):  0.358
FuncNameTest-ruby MRR (bs=1,000):  0.524
Validation-ruby MRR (bs=1,000):  0.396
```

python train.py --model convselfatt ../resources/data/ruby/final/jsonl/train ../resources/data/ruby/final/jsonl/valid ../resources/data/ruby/final/jsonl/test

```

```
