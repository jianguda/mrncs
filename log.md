testrun
2020/02/14 NDCG Average: 0.009783526

### 

neuralbowmodel
rnnmodel
selfattentionmodel
convolutionalmodel
convselfattentionmodel

### 

python train.py --model neuralbow /trained_models ../resources/data/python/final/jsonl/train ../resources/data/python/final/jsonl/valid ../resources/data/python/final/jsonl/test

Test-All MRR (bs=1,000):  0.642
FuncNameTest-All MRR (bs=1,000):  0.524
Validation-All MRR (bs=1,000):  0.608
Test-python MRR (bs=1,000):  0.642
FuncNameTest-python MRR (bs=1,000):  0.524
Validation-python MRR (bs=1,000):  0.608

python train.py --model 1dcnn /trained_models ../resources/data/python/final/jsonl/train ../resources/data/python/final/jsonl/valid ../resources/data/python/final/jsonl/test

Test-All MRR (bs=1,000):  0.515
FuncNameTest-All MRR (bs=1,000):  0.547
Validation-All MRR (bs=1,000):  0.476
Test-python MRR (bs=1,000):  0.515
FuncNameTest-python MRR (bs=1,000):  0.547
Validation-python MRR (bs=1,000):  0.476

python train.py --model convselfatt /trained_models ../resources/data/python/final/jsonl/train ../resources/data/python/final/jsonl/valid ../resources/data/python/final/jsonl/test

...

python train.py --model rnn /trained_models ../resources/data/python/final/jsonl/train ../resources/data/python/final/jsonl/valid ../resources/data/python/final/jsonl/test

Test-All MRR (bs=1,000):  0.635
FuncNameTest-All MRR (bs=1,000):  0.635
Validation-All MRR (bs=1,000):  0.585
Test-python MRR (bs=1,000):  0.635
FuncNameTest-python MRR (bs=1,000):  0.635
Validation-python MRR (bs=1,000):  0.585

python train.py --model selfatt /trained_models ../resources/data/python/final/jsonl/train ../resources/data/python/final/jsonl/valid ../resources/data/python/final/jsonl/test

Test-All MRR (bs=1,000):  0.665
FuncNameTest-All MRR (bs=1,000):  0.630
Validation-All MRR (bs=1,000):  0.621
Test-python MRR (bs=1,000):  0.665
FuncNameTest-python MRR (bs=1,000):  0.630
Validation-python MRR (bs=1,000):  0.621
