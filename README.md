## baselines

**MRR score (train and predict over one language)**

|   Model   |  Go   | Java  |  JS   |  PHP  |  Python   | Ruby  |  Ma-Avg.  |
| :-------: | :---: | :---: | :---: | :---: | :-------: | :---: | :-------: |
| NBOW-raw  | 0.668 | 0.581 | 0.427 | 0.568 | **0.638** | 0.321 | **0.535** |
| 1dCNN-raw | 0.704 | 0.530 | 0.230 | 0.543 |   0.538   | 0.117 |   0.444   |
| biRNN-raw | 0.708 | 0.581 | 0.369 | 0.601 | **0.643** | 0.223 | **0.521** |
| BERT-raw  | 0.726 | 0.551 | 0.417 | 0.601 | **0.677** | 0.351 | **0.554** |

**MRR score (train and predict over all languages)**

|   Model   |  Go   | Java  |  JS   |  PHP  |  Python   | Ruby  |    Avg.    |
| :-------: | :---: | :---: | :---: | :---: | :-------: | :---: | :--------: |
| NBOW-raw  | 0.641 | 0.522 | 0.457 | 0.482 | **0.571** | 0.431 | **0.6170** |
| 1dCNN-raw | 0.636 | 0.523 | 0.352 | 0.538 | **0.577** | 0.258 | **0.6260** |
| biRNN-raw | 0.440 | 0.288 | 0.364 | 0.297 |   0.284   | 0.066 |   0.4280   |
| BERT-raw  |   -   |   -   |   -   |   -   |     -     |   -   |     -      |

MRR: the mean reciprocal rank (MRR) score (the higher the better)

CNN is almost always the worst when it is over single languages, but performs not too badly over multi-languages. In contrast, RNN performs well when it is over single languages, but performs badly over multi-languages. The performance of NBoW is stable and ideal for both two cases. BERT should be very promising as well.

**NDCG score (train and predict over one language)**

|   Model   |  Go   | Java  |  JS   |  PHP  |  Python   | Ruby  |  Ma-Avg.  |
| :-------: | :---: | :---: | :---: | :---: | :-------: | :---: | :-------: |
| NBOW-raw  | 0.117 | 0.209 | 0.065 | 0.149 | **0.292** | 0.129 | **0.161** |
| 1dCNN-raw | 0.014 | 0.116 | 0.010 | 0.124 |   0.204   | 0.040 |   0.085   |
| biRNN-raw | 0.031 | 0.122 | 0.025 | 0.098 |   0.184   | 0.055 |   0.086   |
| BERT-raw  | 0.037 | 0.095 | 0.026 | 0.078 |   0.139   | 0.115 |   0.082   |

**NDCG score (train and predict over all languages)**

|   Model   |  Go   | Java  |  JS   |  PHP  |  Python   | Ruby  |   Avg.    |
| :-------: | :---: | :---: | :---: | :---: | :-------: | :---: | :-------: |
| NBOW-raw  | 0.118 | 0.146 | 0.168 | 0.144 | **0.220** | 0.209 | **0.168** |
| 1dCNN-raw | 0.057 | 0.156 | 0.037 | 0.132 |   0.167   | 0.109 |   0.110   |
| biRNN-raw | 0.040 | 0.081 | 0.014 | 0.060 |   0.065   | 0.039 |   0.050   |
| BERT-raw  |   -   |   -   |   -   |   -   |     -     |   -   |     -     |

NDCG: the Normalized Discounted Cumulative Gain (NDCG) score (the higher the better)

NBoW performs the best, and performances of the other three models are similar to each other but vary on different corpuses.

**Time Cost for Training (Hours)**

|   Model   |  Go   | Java  |  JS   |  PHP  | Python | Ruby  |  All   |
| :-------: | :---: | :---: | :---: | :---: | :----: | :---: | :----: |
| NBOW-raw  | 0.257 | 0.316 | 0.128 | 0.584 | 0.365  | 0.050 | 2.010  |
| 1dCNN-raw | 2.487 | 3.839 | 0.870 | 4.099 | 3.046  | 0.533 | 21.531 |
| biRNN-raw | 1.356 | 3.513 | 0.772 | 2.953 | 3.010  | 0.433 | 18.949 |
| BERT-raw  | 9.544 | 4.558 | 1.072 | 8.850 | 3.497  | 0.536 |   -    |

Overall, NBOW << CNN ≈ RNN < BERT.

## dataset

The number of functions (code snippets along with documents) used for training/validating/testing.

| Data  |   Go    |  Java   |   JS    |   PHP   | Python  |  Ruby  |    All    |
| :---: | :-----: | :-----: | :-----: | :-----: | :-----: | :----: | :-------: |
| Train | 317,832 | 454,451 | 123,889 | 523,712 | 412,178 | 48,791 | 1,880,853 |
| Valid | 14,242  | 15,328  |  8,253  | 26,015  | 23,107  | 2,209  |  89,154   |
| Test  | 14,291  | 26,909  |  6,483  | 28,391  | 22,176  | 2,279  |  100,529  |
|  All  | 346,365 | 496,688 | 138,625 | 578,118 | 457,461 | 53,279 | 2,070,536 |

## About

```markdown
**code** the TensorFlow Implementation
**core** the Keras Implementation
**doc** documentations
----**exp** experimental results
----**setup** how to prepare the Azure VM
----`@csn.md` guidance for run experiments with TensorFlow Implementation
----`@rok.md` guidance for run experiments with Keras Implementation
```

## How to Reproduce Results

### For the TensorFlow implementation

1. prepare the Azure VM by following the TensorFlow1 part at `doc/setup/README`
2. check CodeSearchNet [QuickStart](https://github.com/github/CodeSearchNet#quickstart)
3. override with our implementation by following the guidance at `doc/@csn.md`
4. run the `treeleaf`, `treepath` or `treeraw` model

### For the Keras implementation

1. prepare the Azure VM (following the TensorFlow2 part at `doc/setup/README`)
2. check CodeSnippetSearch [README](https://github.com/novoselrok/codesnippetsearch)
3. override with our implementation by following the guidance at `doc/@rok.md`
4. run models

## References

- [CodeSearchNet](https://github.com/github/CodeSearchNet)
- [CodeSnippetSearch](https://github.com/novoselrok/codesnippetsearch)
