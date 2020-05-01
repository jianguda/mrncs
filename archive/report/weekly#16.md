## DONE

In this week (Week#16),

- For experiments: have run some experiments over tree-path sequence (to get NDCG scores). have tested the multi-modal plan for (leaf-token + tree-path).
- For thesis draft: have added some content on the "experiments" and "results" chapters.

## NEXT-STEPS

1. run experiments with the multi-modal encoder of (leaf-node + tree-path)
2. improve the sub-encoder for tree-path data
3. rerun some experiments
4. compare code2vec with the multi-modal model

## QUESTIONS

None.

## Special Note

The code2vec/code2seq model feeds each tree-path into one LSTM and then concats its tree-path embedding with embeddings of the corresponding leaf-nodes. In my implementation, tree-path data is seen as tokens, just like leaf-node data. In this way, tree-path will not be concated with its leaf-nodes. I am mainly consider the multi-modal model which uses different models for different data, such as NBOW for leaf-token and RNN for tree-path, and then utilize one self-attention layer for different embeddings. In my view, the multi-modal model should perform better than the original code2vec model, because leaf-token data is decoupled from tree-path data. I plan to compare the multi-modal plan with the original code2vec model after completing other experiments.
In this week, I have run some experiments on tree-path data, and their performance is not better than those on leaf-token data. However, I feel these results are still not too bad, and the best NDCG score is 0.189 (by attention-BiRNN) (I have not test CNN or BERT yet), which is very close to 0.190, the best NDCG score on leaf-token data (by attention-BiRNN). Therefore, I believe the combination of leaf-token and tree-path is promising. Meanwhile, I have considered another idea to improve the tree-path idea, which is based on the transformation from arbitrary tree to binary tree. For example, if we transform AST into its left-child right-sibling representation before generating tree-path data, then we reduce the number of possible tree-path and improve the quality of tree-path. In this way, there will be fewer leaf-nodes in the tree and sibling leaf-nodes will be considered together. My idea is from ["Tree-to-tree Neural Networks for Program Translation"](https://arxiv.org/abs/1802.03691) and I feel it is expected to outperform the tree-path idea in code2vec/code2seq. However, I feel it would generate similar data with the hierarchy tree (indent-based tree) so I do not plan to add that option in my implementation, but I am still going to introduce this idea in my report.
In my multi-modal model and the given baseline models, subtokens are not considered. I feel it may affect the tree-path data, also slightly affect the leaf-token data. I will run experiments with using subtokens later.
