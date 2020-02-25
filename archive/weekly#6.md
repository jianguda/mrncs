## DONE
In this week (Week#6), I spent a few days on writting the draft. The first two chapters (introduction, background) and some parts of the third chapter (methodology) are almost ready. Besides,
- For experiments: trained 4/5 baselines on the Python corpus (the hybrid model of 1D-CNN and Self-Attention failed to train because it has more parameters so requires more VRAM). studied the given implementations for baseline models. started hacking the code2seq code.
- For exploration: checked KE WANG's dynamic models, DyPro and LiGer. checked one very recent work, CodeBERT.

## NEXT-STEPS
In next week (Week#7), for baseline models, I will continue the work of integrating code2seq into baseline models as the universal extension work. the MMAN paper uses Tree-LSTM for AST and GGNN for CFG, both Tree-LSTM and GGNN are out-standing models but the whole model seem a bit heavy and complex. Meanwhile, CFG info is not easy to capture. I plan to forget MMAN because its idea exists in code2seq as well.
I considered to make LiGer as the sample of the potential novel model, but its advantages seem not match the contest very well. The another preliminary idea for the novel model is to apply better transformer into popular language models, such as applying reformer into BERT, so I can refer to CodeBERT and try some brute ways of improvements.

## QUESTIONS
None.

## Special Note
None.