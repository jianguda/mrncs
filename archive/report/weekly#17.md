## DONE

In this week (Week#17),

- For experiments: have fixed the implementation and rerun some experiments. have run experiments on data preprocessing, for both code and query. have obtained the second-place NDCG score (0.315) for Python.
- For thesis draft: have added some content on the "experiments" and "results" chapters.

## NEXT-STEPS

1. investigate different loss functions
2. improve the sub-encoder for tree-path data
3. run experiments with the multi-modal encoder of (leaf-node + tree-path)
4. consider other potential improvements

## QUESTIONS

None.

## Special Note

In this week, I firstly spent few days on fixing the original implementation. There are some tiny differences on configurations of different baseline models, besides different code-encoders and query-encoders. Before that, NDCG scores of my extensions are always much worse than NBoW. The former is around 0.2 but the latter is slightly lower than 0.3. I believe it is because of the loss function. The given NBoW model is using the cosine loss but I was using the softmax loss. In five baseline models, only NBoW is using the cosine loss. After the fix work, all my experiments are capable to have similar performance with the given NBoW.
Then I run experiments with my self-attention extension, it can not bring any improvements any more. I did not study the reason because I plan to check that when I am at the multi-modal model. Besides, I also enabled the sub-token option but it greatly reduces the performance, both MRR score and NDCG score. Actually, BPE is enabled by default and it performs much better than sub-token.
In addition, I spent few days on preprocessing work as well, and its experiments are mainly on the source tokens, not on the leaf tokens. For the code-encoder, I tried two plans, one is to remove all keywords and special characters, the other one is to convert all keywords as the token "keyword" and all special characters as the token "character". The NDCG results show that the latter (0.282) is better than the former (0.255) while the given NDCG socre for NBoW model is around 0.27 (but it could be 0.299 at best cases). For the query-encoder, I tried several plans, including, remove 1-length words, remove non-alpha words, remove stop-words, stemming, deduplicate, etc. The best NDCG score I have seen is 0.315 by removing 1-length words and the slightly worse one is 0.303 by removing stop-words. The reason is that, the given queries are usually in high-quality, which can be seen as word query, not phrase query. Besides, I also tried different combinations between these preprocessing methods, but can not find better results. If interested, this is what given query token-sequences look like.
![queries](queries.png)
https://github.com/github/CodeSearchNet/blob/master/resources/queries.csv
In this week, the leader-board shows the first truly awesome competitor, the model named "devout-resonance-31" by roknovosel, which obtained 0.292494426 for the NDCG average and 0.46 for the NDCG score for Python. The notable point is that the first-place model outperforms other models dramatically. Right now, I have three ideas which may improve our results, the first one is to investigate the loss function, and the second one is to try the multi-modal model using leaf-token and tree-path data, and the third one is to make changes the network design (like adding some abstract layers or trying the siamese NN). I plan to try the first two ideas in the following week.
