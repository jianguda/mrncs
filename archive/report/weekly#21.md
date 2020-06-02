## DONE

In this week (Week#21),

- For experiments: have appended LCRS representation. have implemented the desensitization idea. have fixed some bugs so everything is okay. have rerun some experiments.
- For thesis draft: have re-organized the draft. have filled all experiment results. have completed the "results" chapter. have completed about half of other chapters.

## NEXT-STEPS

1. thesis writing
2. thesis presentation

## QUESTIONS

None.

## Special Note

I appended LCRS representation. LCRS is very similar to the SBT representation. It is basically the in-order traversal over the left-child right sibling form of AST/parse tree. Besides, the original code transformation idea is named as "desensitization", because its main purpose is to make all tokens separated from specific programming languages. The concrete implementation is based on string processing, because it is the simplest way. Meanwhile, I have found some bugs hidden in my implementations, so I corrected them, then the attention mechanism start working normally. Until now, I have explored all meaningful directions and also completed its implementation.

I re-organized the whole draft by checking other thesis papers, as well as that thesis report on the similar topic from the university of waterloo. Then, I rerun some experiments and have completed the "results" chapter. In addition, I also have already completed many contents in other chapters. I will turn to the "methodology" chapter and "discussion" chapter. After that, I would polish the whole draft.

According to our results, NBoW + cosine similarity is the best baseline combination. Explorations on three directions bring improvements on accuracy scores. Code representations, especially SBT and LCRS are capable to improve MRR scores. Multi-modal model also guarantees better MRR scores, sometimes better NDCG scores. Exact indexing strategy make sure the most ideal NDCG scores. Meanwhile, we find two promising extensions are not so helpful. Siamese architecture is not better than the given pseudo-siamese architecture. Attention mechanism does not necessarily enhance the code encoder. Last but not least, we have observations on data processing and code desensitization. Data processing does not make sure stable and general improvements, especially for different programming languages. The idea of code desensitization seems good but current implementation does not make obvious positive nor negative effects on results.
