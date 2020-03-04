## DONE
In this week (Week#8),
- For experiments: have improved the pre-processing scripts. have refactored the implementation of the modified code2seq (to reduce the difficulty of integration, also help understand its code). have manually checked the modified code2seq.
- For exploration: have checked GNNs and the lib to use. My work will start from reproducing the encoder from this [repo](https://github.com/microsoft/graph-based-code-modelling).
- For thesis draft: add some content to "background" and "discussion".

## NEXT-STEPS
1) generate ASTs and Paths of Python corpus and test the modified code2seq
2) make the modified code2seq as standalone code encoder in the CSN framework and run experiments
3) try to make the modified code2seq work in parallel with current code encoders
4) write some contents in my paper draft, if I have time, check the graph encoder

## QUESTIONS
None.

## Special Note
I am using the Python-specified ast parser. I plan to run experiments on Python corpus firstly. I will turn to use the tree-sitter ast parser for other corpuses (I guess maybe after the GGNN model).
