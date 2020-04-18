## baselines

**MRR score (train over single language and predict over that language)**

|   Model   |  Go   | Java  |  JS   |  PHP  |  Python   | Ruby  |  Ma-Avg.  |
| :-------: | :---: | :---: | :---: | :---: | :-------: | :---: | :-------: |
| NBOW-raw  | 0.668 | 0.581 | 0.427 | 0.568 | **0.643** | 0.321 | **0.535** |
| 1dCNN-raw | 0.704 | 0.530 | 0.230 | 0.543 |   0.538   | 0.117 |   0.444   |
| biRNN-raw | 0.708 | 0.581 | 0.369 | 0.601 | **0.643** | 0.223 | **0.521** |
| BERT-raw  | 0.726 | 0.551 | 0.417 | 0.601 | **0.677** | 0.351 | **0.554** |

**MRR score (train over all language and predict over all languages)**

|   Model   |  Go   | Java  |  JS   |  PHP  |  Python   | Ruby  |    Avg.    |
| :-------: | :---: | :---: | :---: | :---: | :-------: | :---: | :--------: |
| NBOW-raw  | 0.641 | 0.522 | 0.457 | 0.482 | **0.571** | 0.431 | **0.6170** |
| 1dCNN-raw | 0.636 | 0.523 | 0.352 | 0.538 | **0.577** | 0.258 | **0.6260** |
| biRNN-raw | 0.440 | 0.288 | 0.364 | 0.297 |   0.284   | 0.066 |   0.4280   |
| BERT-raw  |   -   |   -   |   -   |   -   |     -     |   -   |     -      |

MRR: the mean reciprocal rank (MRR) score (the higher the better)

CNN is almost always the worst when it is over single languages, but performs not too badly over multi-languages. In contrast, RNN performs well when it is over single languages, but performs badly over multi-languages. The performance of NBOW is stable and ideal for both two cases. BERT should be very promising as well.

**NDCG score (train over single language and predict over that language)**

|   Model   |  Go   | Java  |  JS   |  PHP  |  Python   | Ruby  |  Ma-Avg.  |
| :-------: | :---: | :---: | :---: | :---: | :-------: | :---: | :-------: |
| NBOW-raw  | 0.117 | 0.209 | 0.065 | 0.149 | **0.299** | 0.129 | **0.161** |
| 1dCNN-raw | 0.014 | 0.116 | 0.010 | 0.124 |   0.204   | 0.040 |   0.085   |
| biRNN-raw | 0.031 | 0.122 | 0.025 | 0.098 |   0.184   | 0.055 |   0.086   |
| BERT-raw  | 0.037 | 0.095 | 0.026 | 0.078 |   0.139   | 0.115 |   0.082   |

**NDCG score (train over all language and predict over all languages)**

|   Model   |   Go    |  Java   |   JS    |   PHP   | Python  |  Ruby   |    Avg.     |
| :-------: | :-----: | :-----: | :-----: | :-----: | :-----: | :-----: | :---------: |
| NBOW-raw  |    -    |    -    |    -    |    -    |    -    |    -    | 0.167579375 |
| 1dCNN-raw | 0.05674 | 0.15570 | 0.03733 | 0.13210 | 0.16670 | 0.10920 | 0.109630036 |
| biRNN-raw | 0.04016 | 0.08116 | 0.01377 | 0.05979 | 0.06464 | 0.03939 | 0.049820371 |
| BERT-raw  |    -    |    -    |    -    |    -    |    -    |    -    |      -      |

NDCG: the Normalized Discounted Cumulative Gain (NDCG) score (the higher the better)

NBOW performs the best, and performances of the other three models are similar to each other but vary on different corpuses.

**Time Cost for Training (Hours)**

|   Model   |  Go   | Java  |  JS   |  PHP  | Python | Ruby  |  All   |
| :-------: | :---: | :---: | :---: | :---: | :----: | :---: | :----: |
| NBOW-raw  | 0.257 | 0.316 | 0.128 | 0.584 | 0.392  | 0.050 | 2.010  |
| 1dCNN-raw | 2.487 | 3.839 | 0.870 | 4.099 | 3.046  | 0.533 | 21.531 |
| biRNN-raw | 1.356 | 3.513 | 0.772 | 2.953 | 3.010  | 0.433 | 18.949 |
| BERT-raw  | 9.544 | 4.558 | 1.072 | 8.850 | 3.497  | 0.536 |   -    |

Overall, NBOW << CNN ≈ RNN < BERT.

## extensions

**based on leaf-token data, train over Python and predict over Python**

|    Model     |    MRR    |   NDCG    |
| :----------: | :-------: | :-------: |
|  NBOW-leaf   | **0.532** | **0.132** |
| attNBOW-leaf | **0.540** | **0.142** |
|   CNN-leaf   |   0.276   |   0.077   |
| attCNN-leaf  |   0.208   |   0.043   |
|   RNN-leaf   | **0.655** | **0.183** |
| attRNN-leaf  | **0.655** | **0.190** |
|  BERT-leaf   |   0.429   |   0.067   |
| attBERT-leaf |   0.099   |   0.004   |

**based on leaf-token + keyword data, train over Python and predict over Python**

for different types of tree-path data

|    Model     | MRR | NDCG |
| :----------: | :-: | :--: |
|  NBOW-path   |  -  |  -   |
| attNBOW-path |  -  |  -   |
|   RNN-path   |  -  |  -   |
| attRNN-path  |  -  |  -   |

**based on tree-path data, train over Python and predict over Python**

for different types of tree-path data

|    Model     | MRR | NDCG |
| :----------: | :-: | :--: |
|  NBOW-path   |  -  |  -   |
| attNBOW-path |  -  |  -   |
|   RNN-path   |  -  |  -   |
| attRNN-path  |  -  |  -   |

**based on hybrid data (leaf-token + tree-path), train over Python and predict over Python**

|         Model          | MRR | NDCG |
| :--------------------: | :-: | :--: |
| Tree-hybrid (NBOW+RNN) |  -  |  -   |

## dataset

The number of functions (code snippets along with documents) used for training/validating/testing.

| Data  |   Go   |  Java  |   JS   |  PHP   | Python | Ruby  |   All   |
| :---: | :----: | :----: | :----: | :----: | :----: | :---: | :-----: |
| Train | 317832 | 454451 | 123889 | 523712 | 412178 | 48791 | 1880853 |
| Valid | 14242  | 15328  |  8253  | 26015  | 23107  | 2209  |  89154  |
| Test  | 14291  | 26909  |  6483  | 28391  | 22176  | 2279  | 100529  |
|  All  | 346365 | 496688 | 138625 | 578118 | 457461 | 53279 | 2070536 |

## about this repo

```markdown
**archive** documentations

- **exp** experiment raw data
- **report** weekly progress report
- **setup** how to setup the Azure VM
- `@.md` sensitive info
- `memo.md` links of reference materials

**code** changes to CodeSearchNet code

- **encoders** ...
- **models** ...
- **scripts** scripts for generating ASTs, Paths and Graphs
```

## how to run

1. prepare the Azure VM (following `archive/setup`)
2. check CodeSearchNet [QuickStart](https://github.com/github/CodeSearchNet#quickstart)
3. override the official **code** with mine
4. run the `alon` model

## changes

compared with the CSN code, my changes are on following code files:

- `code/encoders/__init__.py`
- `code/encoders/alon_encoder.py`
- `code/encoders/seq_encoder.py`
- `code/encoders/tmp_encoder.py`
- `code/encoders/tree_tmp_encoder.py`
- `code/encoders/encoder.py`
- `code/models/__init__.py`
- `code/models/alon_model.py`
- `code/models/tree_model.py`
- `code/models/model.py`
- `code/scripts/*`
- `code/model_restore_helper.py`
- `code/predict.py`
- `code/train.py`
- `code/wow.py`
