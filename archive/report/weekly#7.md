## DONE
In this week (Week#7),
- For experiments: have reproduced all baseline models on all six corpuses (excluding the the CNN+attention hybrid model). have completed the modifications on code2seq (by replacing the decoder with attention weighted embedding). have completed scripts for generating ASTs, extracting and preprocessing Paths.
- For exploration: checked one paper discussing about GNNs. checked some blogs connecting transformers and GNNs. determined the novel model being one GNN model instead of one transformer model.

## NEXT-STEPS
In next week (Week#8), i will generate the Path data for Python corpus, and then integrate code2seq as one encoder in the CSN framework. There are two ways to add this encoder to current models, one is to replace current code encoders, and the other one is to make it work in parallel with current code encoders. I plan to check both of them on Python corpus. Meanwhile, I will plan the concrete work for the novel model.

## QUESTIONS
None.

## Special Note
Among baseline models, I decided to ignore the hybrid model based on following reasons: 1) it is ignored as well at CodeBERT's experiment section. 2) CNN always performs the worst no matter on which corpus, maybe CNN+attention still won't work very well? 3) my work has weak connections with this hybrid model :)
