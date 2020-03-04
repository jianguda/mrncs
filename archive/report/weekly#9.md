## DONE
In this week (Week#9),
- For experiments: have generated ASTs and Paths on Python corpus. have implemented the basic tree-sitter parser. have checked some code to integrate the modified version of code2seq.
- For thesis draft: have added lots of content on "background", have added some content on "methods".

## NEXT-STEPS
1) integrate code2seq as soon as possible
2) improve the tree-sitter parser to reduce the failing cases
3) add some content on path-based representation in the paper draft

## QUESTIONS
None.

## Special Note
1) the generated data is not prefect, the ratio of failing cases:
45827/412178 (train-set) 2679/23107 (valid-set) 1705/22176 (test-set)
but I already have the data for experiments! it is harder to improve the py150k parser so I turned to implement corresponding tree-sitter parsers, which can easily support all languages
2) my next step is to integrate code2seq and run experiments so get the basic results. I want to devote more time on path-based methods. Even though I have the willing to experiment the GGNN plan, but i have to restrain myself and keep my work plan being realistic
3) because of the corona-virus, I decide to stay at home as much as possible from next week
