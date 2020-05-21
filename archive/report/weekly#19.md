## DONE

In this week (Week#19),

- For experiments: have tried the KNN-based indexing strategy. have supported multi-modal model in Keras implementation. have fixed and improved the Keras implementation.
- For thesis draft: I mainly spent time on experiments recently so I did not do thesis writing.

## NEXT-STEPS

1. improve the accuracy of Keras implementation
2. better data preprocessing (see SPECIAL NOTE for more details)
3. support other AST representation (like SBT)
4. try different strategies for multi-modal learning
5. try more other models as the code encoder

## QUESTIONS

None.

## Special Note

In this week, inspired from Rok's solution, I compared different indexing strategies and obtained very nice results (temporarily ranked first at the leaderboard) by applying KNN. With the reference to Rok's implementation, I have completed my Keras implementation and have supported more models. Right now, I have both the implementation over the given baseline models and the Keras implementation. I prefer the latter because it takes less time on developing and experimenting.

I have noticed that Keras implementation actually greatly reduced the accuracy of the given baseline models, so do Rok's implementation. For example, in Rok's implementation, when using Annoy Index, for the NDCG score over the Python corpus, the former is around 0.18 but the latter is around 0.29. I have fixed some small issues in my Keras implementation but the improved accuracy is still not very ideal. Therefore, I will continue working for eliminating this "accuracy gap". Maybe I will find some important factors which may bring higher accuracy in this direction.

In addition, I will consider other ideas for improvements. I have noticed one phenomenon that, the NDCG scores by models trained over all 6 languages are not necessarily better than models trained over only single language. For 6 languages in the Corpus, for the data amount, PHP > Java > Python > Go >> JS > Ruby. Based on my experiments on the given baseline models, for Java and Python, NBoW over the single language corpus performs better, but for JS and Ruby, NBoW over the whole corpus performs better. For PHP and Go, both models have similar performance. This phenomenon also exists when we are using KNN indexing (NDCG for NBoW model trained over Python could be 0.499 but for NBoW model trained over the whole corpus is 0.457). I believe that in the most ideal cases, NDCG scores of models trained over the whole corpus should alway be better than those trained over only single language corpus, like what happened over JS and Ruby. Besides, I believe that it is because of the quality difference between code inputs and query inputs. My guess is that the code-query pairs in Java or Python are with better quality. Meanwhile, the absolute accuracy value between Java and Python indicates that Python is more ideal than Java and the most ideal language over all 6 languages, for this model framework or this task. The most direct and simplest solution is to convert all other languages to Python (in that case, the mean NDCG scores would be around 0.5), unless it is just because queries for other languages are with lower quality of varying extents. Therefore, I will do some explorations in this direction as well.

I have supported the multi-modal model on Keras implementation. Right now, the strategy for multi-modal learning is very simple, namely directly catenating two embeddings. I plan to try other strategies as well. Meanwhile, I will implement other AST representations, such as SBT. In addition, I will try more other models as the code encoder, because my thesis topic has changed to "A Large-scale Experimental Comparison of Machine-learning Based Code Search on the CodeSearchNet Corpus" from "Semantic Code Search via Tree-Path Representation".
