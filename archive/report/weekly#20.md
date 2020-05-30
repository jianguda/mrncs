## DONE

In this week (Week#20),

- For experiments: have improved Keras implementation and fixed bugs. have tried two multi-modal models (leaf+path, code+sbt). have tried self-attention in encoders.
- For thesis draft: have updated some experiment results. have added some contents on "background" and "conclusion".

## NEXT-STEPS

1. thesis draft writing
2. compare different tree-paths
3. check attention implementation
4. complete code transformation idea

## QUESTIONS

None.

## Special Note

The reason why the Keras implementation performs worse than the official implementation is because, the batch size seems too large. When batch-size is too large, the model may be at the local optimum. Besides, it also works if we just append Dropout layers at encoders.

I have tried two multi-modal models, one is over code tokens and sbt representation, and the other one is over tree-leaf tokens and tree-leaf sequence. They did not bring too much improvements. For the NDCG score over Python corpus, the former increased to 0.427 from 0.420, the latter decreased slightly. Because there are different types of tree-paths, I need to run some experiments for their comparisons.

Based on some papers on image-text matching, such as [SCAN](https://arxiv.org/abs/1803.08024) and [MHSAN](https://arxiv.org/abs/2001.03712), I have appended self-attention mechanism in encoders but its result seems not very ideal. I will recheck my implementation.

In this week, the challenge organizer updated the results for baseline models by updating the indexing strategy, and the vanilla NBoW model, as the best baseline model, is ranked the second place. I have not yet completed the code translation idea because I feel preprocessing work on code tokens is doing the same thing. I have two ways, one is directly improve the existing preprocessing work and other one is based on parse trees. I am going to run experiments over different languages, and then decide how to complete that idea. In ideal case, NDCG scores on other languages should be close to that on Python corpus.

Until now, I mainly explored ideas over these directions: indexing strategy, preprocessing, data-enhancement, siamese architecture, multi-modal models, attention-aided encoders. According to my results, the first two options are most helpful. As planned, I will mainly work for the thesis writing from next week. Considering I have completed many contents, I am going to complete the draft at next two weeks. Once I have completed some chapters, I would send emails to @Zimin.
