## DONE

In this week (Week#12),

- For experiments: have improved the TS-based path-extractor now it can parse all data without error. have generated the whole ast-path data for the Python corpus. have tried to run some experiments on the whole ast-path data but failed because of strange memory error.
- For thesis draft: have added some contents to "methods" chapter. have made some changes to "introduction" and "background" chapter based on Zimin's comments.

## NEXT-STEPS

1. handle the memory error (maybe I just need to switch the way of loading data, but not sure)
2. run experiments on the alon-encoder to make sure the model is okay
3. improve the model implementation if results are not as good as baselines
4. improve the implementation of path-extractor to support SPT, CAPT, as well as DFSud, Leaf2LeafUD
5. consider the concrete specification of "Hierarchy Parse Tree"
6. add more contents to the "methods" chapter and "results" chapter

## QUESTIONS

None.

## Special Note

In baseline models, code encoders are mainly for token sequences. My original experiment setting is to apply Alon's AST-path sequences for semantic code search. It aims to compare semantic encoder and syntactic encoder. Based on [CAPT](https://arxiv.org/abs/2003.11118), I feel it is meaningful to compare different tree-structured representation. Besides, from [this paper](https://arxiv.org/abs/2003.13848), I feel it is meaningful to compare different way for generating tree-path sequences as well. Therefore, I plan to add these comparisons to my experiment settings.

Besides, starting from the indent-based tree in [TranS^3](https://arxiv.org/abs/2003.03238), I plan to have one general idea like "Hierarchy Parse Tree" and it will in the comparison with AST, Aroma's SPT, CAPT (the default setting is the same as SPT, so I will consider another best setting). There have been two plans for tree-path sequences, namely DFSud, Leaf2Leaf (Alon's Tree-path embedding plan). I plan to add one option "Leaf2LeafUD". It add the direction info (upper, down) to Leaf2Leaf path and can be seen as the hybrid version of DFSud and Leaf2Leaf. Therefore, there will be 12 different designs of Path-based Encoder.
