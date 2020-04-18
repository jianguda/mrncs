## DONE

In this week (Week#15),

- For experiments: have moved to new implementation. have run all runnable experiments over leaf-node sequences (to get NDCG scores).
- For thesis draft: have made some small changes.

## NEXT-STEPS

1. run experiments on different types of tree-path
1. run experiments over (leaf-nodes + keywords)
1. run experiments over (leaf-nodes + tree-paths)

## QUESTIONS

None.

## Special Note

Right now, the current implementation is to reuse existing code in CodeSearchNet repo and only make necessary changes (append attention module, combine parallel encoders, etc), instead of integrating the modified version of code2vec implementation. There will be some changes to the implementation as well but it is still based on existing code. In this way, I can make sure all code are valid and avoid debugging NN.
In this week, I figured out one way to compute NDCG scores without submitting to the leaderboard in this week, so I re-ran some baseline experiments. There are two kinds of experiments, one is to train models over single language and predict over that language, and the other one is to train models over multi-languages. We mainly take care of the former, and choose Python as the objective. According to the [results](https://github.com/jianguda/csn#baselines), NBOW is the most outstanding model, and its NDCG score can be **0.299**, which can be the second best result at the leaderboard (the current first place is **0.302**)! Meanwhile, we can conclude that it is obvious that better MRR score does not mean better NDCG score. Therefore, I plan to improve tree-model mainly based on NDCG scores.
Compare [results](https://github.com/jianguda/csn#extensions) of leaf-tokens with that of raw data, we can see RNN is better in MRR and slightly better in NDCG. I believe it is because RNN can not focus on important tokens when it is trained over raw data. Another observation is that, NBOW performs worse comparing with that of raw data. I feel it actually indicates that non-terminals in AST are important as well, my original guess is that, structural information is meaningful, but I have another guess is that non-terminals could be informative just by themselves.
Because NBOW and RNN are the most promising baseline models, the further experiments like (leaf-token + keywords) and (leaf-token + tree-path) will only be conducted with them. There are two ways of utilizing keywords in my mind, one is to see it as sequences so I can append tree-paths later, and another idea is to use tree-LSTM (no tree-path in that way). I will mainly or firstly consider the first option.
