## baselines

**MRR score (train over single language and predict over that language)**

|   Model   |  Go   | Java  |  JS   |  PHP  |  Python   | Ruby  |  Ma-Avg.  |
| :-------: | :---: | :---: | :---: | :---: | :-------: | :---: | :-------: |
| NBOW-raw  | 0.668 | 0.581 | 0.427 | 0.568 | **0.638** | 0.321 | **0.535** |
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
| NBOW-raw  | 0.117 | 0.209 | 0.065 | 0.149 | **0.292** | 0.129 | **0.161** |
| 1dCNN-raw | 0.014 | 0.116 | 0.010 | 0.124 |   0.204   | 0.040 |   0.085   |
| biRNN-raw | 0.031 | 0.122 | 0.025 | 0.098 |   0.184   | 0.055 |   0.086   |
| BERT-raw  | 0.037 | 0.095 | 0.026 | 0.078 |   0.139   | 0.115 |   0.082   |

**NDCG score (train over all language and predict over all languages)**

|   Model   |  Go   | Java  |  JS   |  PHP  |  Python   | Ruby  |   Avg.    |
| :-------: | :---: | :---: | :---: | :---: | :-------: | :---: | :-------: |
| NBOW-raw  | 0.118 | 0.146 | 0.168 | 0.144 | **0.220** | 0.209 | **0.168** |
| 1dCNN-raw | 0.057 | 0.156 | 0.037 | 0.132 |   0.167   | 0.109 |   0.110   |
| biRNN-raw | 0.040 | 0.081 | 0.014 | 0.060 |   0.065   | 0.039 |   0.050   |
| BERT-raw  |   -   |   -   |   -   |   -   |     -     |   -   |     -     |

NDCG: the Normalized Discounted Cumulative Gain (NDCG) score (the higher the better)

NBOW performs the best, and performances of the other three models are similar to each other but vary on different corpuses.

**Time Cost for Training (Hours)**

|   Model   |  Go   | Java  |  JS   |  PHP  | Python | Ruby  |  All   |
| :-------: | :---: | :---: | :---: | :---: | :----: | :---: | :----: |
| NBOW-raw  | 0.257 | 0.316 | 0.128 | 0.584 | 0.365  | 0.050 | 2.010  |
| 1dCNN-raw | 2.487 | 3.839 | 0.870 | 4.099 | 3.046  | 0.533 | 21.531 |
| biRNN-raw | 1.356 | 3.513 | 0.772 | 2.953 | 3.010  | 0.433 | 18.949 |
| BERT-raw  | 9.544 | 4.558 | 1.072 | 8.850 | 3.497  | 0.536 |   -    |

Overall, NBOW << CNN ≈ RNN < BERT.

## extensions

|       Model        | MRR (Python) | NDCG (Python) | MRR (all) | NDCG (all) |
| :----------------: | :----------: | :-----------: | --------- | ---------- |
|        NBoW        |    0.638     |     0.292     | 0.571     | 0.220      |
| NBoW-Preprocessing |    0.643     |     0.312     | -         | -          |
|      NBoW-KNN      |    0.643     |   **0.499**   | 0.620     | **0.370**  |

<!-- **based on leaf-token data, train over Python and predict over Python**

RNN is used as the query encoder

|    Model     |    MRR    |   NDCG    |
| :----------: | :-------: | :-------: |
|  NBOW-leaf   | **0.530** | **0.152** |
| attNBOW-leaf | **0.539** | **0.156** |
|   CNN-leaf   |   0.276   |   0.077   |
| attCNN-leaf  |   0.257   |   0.087   |
|   RNN-leaf   | **0.666** | **0.171** |
| attRNN-leaf  | **0.653** | **0.180** |
|  BERT-leaf   |   0.429   |   0.067   |
| attBERT-leaf |   0.099   |   0.004   | -->

<!-- |  Tree-leaf   |   0.531   |   0.129   |
| attTree-leaf |   0.008   |   0.005   | -->

<!-- **based on leaf-token data, train over Python and predict over Python**

NBOW is used as the query encoder

|    Model     |    MRR    |   NDCG    |
| :----------: | :-------: | :-------: |
|  NBOW-leaf   | **0.587** | **0.167** |
| attNBOW-leaf | **0.579** | **0.201** |
|   CNN-leaf   |   0.209   |   0.141   |
| attCNN-leaf  |   0.137   |   0.102   |
|   RNN-leaf   | **0.588** | **0.186** |
| attRNN-leaf  | **0.440** | **0.185** |
|  BERT-leaf   |   0.508   |   0.156   |
| attBERT-leaf |   0.481   |   0.126   | -->

<!-- **conclusion** we have run some experiments with the given four baseline models and found NBOW and RNN (+attention) are expected to generate the most ideal NDCG scores. Besides, we compared the two cases where RNN and NBOW are used as the query encoder, and find NDCG scores are always higher when we use NBOW query-encoder. Therefore, in later experiments, we only NBOW as query-encoder, and try NBOW and RNN (attention) for code encoder. -->

---

because NBOW is the best baseline model, we make improvements on the NBOW model. Which means, NBOW is used as the coder encoder and the query encoder.

**NBOW train over Python and predict over Python**

|               Model                |  MRR  |   NDCG    |
| :--------------------------------: | :---: | :-------: |
|                raw                 | 0.638 |   0.292   |
|         raw-preprocessing          | 0.643 | **0.312** |
|                leaf                | 0.620 |   0.267   |
| leaf-preprocessing(non-len1-words) | 0.616 |   0.242   |
| leaf-preprocessing(non-stop-words) | 0.612 |   0.259   |
|                path                |   -   |     -     |
|         path-preprocessing         |   -   |     -     |
|                tree                |   -   |     -     |
|           tree-attention           |   -   |     -     |
|         tree-preprocessing         |   -   |     -     |

<!-- |    raw-subtoken    | 0.015 | 0.003 | -->

**NBOW train over Python and predict over Python**

|                                Model                                |  MRR  |   NDCG    |
| :-----------------------------------------------------------------: | :---: | :-------: |
|                                 raw                                 | 0.638 |   0.292   |
|                 raw-preprocessing(convert)(normal)                  | 0.645 |   0.282   |
|                 raw-preprocessing(discard)(normal)                  | 0.646 |   0.255   |
|              raw-preprocessing(normal)(non-len1-words)              | 0.636 | **0.315** |
|              raw-preprocessing(normal)(non-len2-words)              | 0.636 |   0.251   |
|              raw-preprocessing(normal)(non-len3-words)              | 0.607 |   0.226   |
|                raw-preprocessing(normal)(non-digit)                 | 0.639 |   0.265   |
|             raw-preprocessing(normal)(non-punctuation)              | 0.643 |   0.280   |
|        raw-preprocessing(normal)(non-digit&non-punctuation)         | 0.641 |   0.291   |
|                raw-preprocessing(normal)(only-alpha)                | 0.611 |   0.263   |
|              raw-preprocessing(normal)(non-stop-words)              | 0.643 | **0.312** |
|                 raw-preprocessing(normal)(stemming)                 | 0.638 |   0.256   |
|               raw-preprocessing(normal)(deduplicate)                | 0.638 |   0.285   |
|      raw-preprocessing(normal)(non-len1-words&non-stop-words)       | 0.640 |   0.291   |
|      raw-preprocessing(normal)(non-punctuation&non-stop-words)      | 0.637 |   0.265   |
| raw-preprocessing(normal)(non-digit&non-punctuation&non-stop-words) | 0.638 |   0.255   |

**NBOW train over Python and predict over Python**

|       Model       |    MRR    |   NDCG    | time (hours) |
| :---------------: | :-------: | :-------: | :----------: |
|   raw(softmax)    |   0.582   |   0.173   |    0.360     |
| raw(cosine-annoy) | **0.638** | **0.292** |    0.365     |
|  raw(cosine-KNN)  | **0.643** | **0.499** |    0.403     |
|  raw(max-margin)  |   0.583   |   0.176   |    0.355     |
|   raw(triplet)    |   0.611   |   0.266   |    1.603     |

<!-- |   CNN(cosine)   |   0.008   |   0.042   |    1.025     |
|   RNN(cosine)   |   0.008   |   0.066   |    2.117     |
|  BERT(cosine)   |     -     |     -     |      -       | -->

**NBOW train over Python and predict over Python**

|         Model          |  MRR  | NDCG  |
| :--------------------: | :---: | :---: |
|        treeleaf        | 0.620 | 0.267 |
|   treepath(AST+L2L)    |   -   |   -   |
|   treepath(SPT+L2L)    |   -   |   -   |
|   treepath(HST+L2L)    |   -   |   -   |
|   treepath(HPT+L2L)    |   -   |   -   |
|  treepath(AST/SPT+UD)  |   -   |   -   |
|  treepath(HST/HPT+UD)  | 0.032 | 0.003 |
|   treepath(AST+U2D)    |   -   |   -   |
|   treepath(SPT+U2D)    |   -   |   -   |
|   treepath(HST+U2D)    |   -   |   -   |
|   treepath(HPT+U2D)    |   -   |   -   |
|  code2vec(leaf+path)   |   -   |   -   |
| multi-modal(leaf+path) |  ...  |  ...  |

**conclusion** ...

## start from Rok

|    Model     | MRR (Python) | NDCG (Python) | MRR (all) | NDCG (all) |
| :----------: | :----------: | :-----------: | --------- | ---------- |
|   vanilla    |    0.637     |     0.437     | 0.696     | 0.287      |
| siamese-code |    0.008     |     0.000     | -         | -          |
| siamese-leaf |    0.008     |     0.000     | -         | -          |

## index

annoy: approximate nearest neighbor
KNN: exact nearest neighbor

|      Model       |    MRR    |   NDCG    |
| :--------------: | :-------: | :-------: |
|   nbow(annoy)    | **0.638** | **0.292** |
|    nbow(KNN)     | **0.643** | **0.499** |
| rok(annoy) - 128 |     -     |   0.218   |
|  rok(KNN) - 128  |     -     |   0.409   |
| rok(annoy) - 256 |     -     |   0.184   |
|  rok(KNN) - 256  |     -     |   0.426   |

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
- `@csn.md` sensitive info and guidance for csn (code1)
- `@rok.md` sensitive info and guidance for rok (code2)
- `memo.md` links of reference materials

**code** changes to CodeSearchNet code

- **encoders** ...
- **models** ...
- **scripts** scripts for generating ASTs, Paths and Graphs

**core** the "devout-resonance-31" model by Rok Novosel https://github.com/novoselrok/codesnippetsearch

- **rok**: changes to the original **code_search** (process data, train models and cache embeddings)
```

## how to run

1. prepare the Azure VM (following `archive/setup`)
2. check CodeSearchNet [QuickStart](https://github.com/github/CodeSearchNet#quickstart)
3. override the official **src** folder with my **code** folder
4. run the `treeleaf`, `treepath`, `treeraw` or `treeall` model
5. check CodeSnippetSearch [README](https://github.com/novoselrok/codesnippetsearch)
6. override the official **code_search** folder with my **better** folder
7. run models

## changes

compared with the CSN code, my changes are on following code files:

- `code/encoders/__init__.py`
- `code/encoders/encoder.py`
- `code/encoders/seq_encoder.py`
- `code/encoders/tree/*`
- `code/models/__init__.py`
- `code/models/alon_model.py`
- `code/models/tree_all_model.py`
- `code/models/tree_leaf_model.py`
- `code/models/tree_path_model.py`
- `code/models/tree_raw_model.py`
- `code/models/model.py`
- `code/scripts/*`
- `code/model_restore_helper.py`
- `code/predict.py`
- `code/train.py`
- `code/wow.py`
