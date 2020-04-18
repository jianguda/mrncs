## DONE

In this week (Week#14),

- For experiments: have run some experiments over leaf-node sequences. have made some progress in debugging model.
- For thesis draft: have made some small changes.

## NEXT-STEPS

1. complete the debug work of model implementation
2. run experiments on leaf-node and different tree-path data
3. run experiments with different plans of (leaf-node + tree-path)

## QUESTIONS

None.

## Special Note

I feel the combination of leaf-node and tree-path should be more meaningful than just tree-path, so this week I have run experiments on leaf-node sequences. Only identifiers are considered because I feel they have contained the most important semantic info. With the given biRNN model, the MRR score on Python corpus is 0.663, which is better than all baseline models (best MRR score is 0.621 by given BERT model).
I feel it is a bit hard to debug the current implementation, so I turned to modify the given RNN model step by step until I get the target model. In this way, I can validate my model from time to time. Right now, I am at the step of adding the given self-attention module to RNN, and it feels like the given CNN-self-attention. However, I meet the GPU memory issue (too many parameters in the model, and it happened as well when I was trying to run CNN-self-attention before). I plan to use the self-attention module used in the code2vec implementation.
I feel I can get more results at next week.
