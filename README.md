## baselines

|Model  |Go     |Java   |JS     |PHP    |Python |Ruby   |
|:-:    |:-:    |:-:    |:-:    |:-:    |:-:    |:-:    |
|NBOW   |0.802  |0.572  |0.415  |0.576  |0.608  |0.357  |
|1dCNN  |0.785  |0.498  |0.232  |0.537  |0.476  |0.118  |
|biRNN  |0.787  |0.562  |0.364  |0.600  |0.585  |0.242  |
|BERT   |0.822  |0.531  |0.414  |0.597  |0.621  |0.396  |
|Hybrid | -     | -     | -     | -     | -     | -     |

MRR: the mean reciprocal rank (MRR) score (the higher the better)

__computation time__
According to my memory, NBOW << CNN < RNN << BERT.

__performance__
CNN is always the worst, the other three models are all competitive but perform differently on different corpuses.

## about this repo
```markdown
__archive__ documentations
+ __exp__ experiment raw data
+ __log__ weekly progress report
+ __setup__ how to setup the Azure VM
+ `@.md` sensitive info
+ `memo.md` links of reference materials
__code__ changes to CodeSearchNet code
+ __encoders__ ...
+ __models__ ...
+ __scripts__ scripts for generating ASTs, Paths and Graphs
```
