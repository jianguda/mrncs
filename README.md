## baselines

| Model  |  Go   | Java  |  JS   |  PHP  | Python | Ruby  |  Avg.  |
| :----: | :---: | :---: | :---: | :---: | :----: | :---: | :----: |
|  NBOW  | 0.802 | 0.572 | 0.415 | 0.576 | 0.608  | 0.357 | 0.5550 |
| 1dCNN  | 0.785 | 0.498 | 0.232 | 0.537 | 0.476  | 0.118 | 0.4410 |
| biRNN  | 0.787 | 0.562 | 0.364 | 0.600 | 0.585  | 0.242 | 0.5233 |
|  BERT  | 0.822 | 0.531 | 0.414 | 0.597 | 0.621  | 0.396 | 0.5635 |
| Hybrid |   -   |   -   |   -   |   -   |   -    |   -   |   -    |

MRR: the mean reciprocal rank (MRR) score (the higher the better)

| Model  | Go  | Java | JS  | PHP | Python | Ruby |    Avg.     |
| :----: | :-: | :--: | :-: | :-: | :----: | :--: | :---------: |
|  NBOW  |  -  |  -   |  -  |  -  |   -    |  -   |      -      |
| 1dCNN  |  -  |  -   |  -  |  -  |   -    |  -   |      -      |
| biRNN  |  -  |  -   |  -  |  -  |   -    |  -   | 0.049820371 |
|  BERT  |  -  |  -   |  -  |  -  |   -    |  -   |      -      |
| Hybrid |  -  |  -   |  -  |  -  |   -    |  -   |      -      |

NDCG: the Normalized Discounted Cumulative Gain (NDCG) score (the higher the better)

**computation time**
According to my memory, NBOW << CNN < RNN << BERT.

**performance**
CNN is always the worst, the other three models are all competitive but perform differently on different corpuses.

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
- **log** experiment raw data (new)
- **report** weekly progress report
- **setup** how to setup the Azure VM
- `@.md` sensitive info
- `memo.md` links of reference materials
  **code** changes to CodeSearchNet code
- **encoders** ...
- **models** ...
- **scripts** scripts for generating ASTs, Paths and Graphs
```
