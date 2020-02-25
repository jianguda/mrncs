## DONE
In this week (Week#4), I have read a few blogs and papers about embedding, transformer and other topics, mainly to explore possible directions for my work. I also have checked some high-quality master thesises in DiVA database and started my writting. Meanwhile, I spent some time in preparing the experiment environment.

As planned, my work mainly contains two parts, one is to extend given baseline models, and the other one is to propose one possible novel model. For the first part, based on this paper "Multi-Modal Attention Network Learning for Semantic Source Code Retrieval", my current idea is to combine code tokens and program AST. The original paper is about "tokens + AST + CFG", but according to its claimed results, CFG model seems only bring slight improvements. Both AST and CFG models are expected to help capture features by introducing prior knowledge. For the second part, the preliminary plan is to adapt one powerful feature extractor.

I have filled some parts in the first two chapters (Introduction, Background) of my thesis. I will add more contents to these chapters gradually.

## NEXT-STEPS
In next week (Week#5), I will reproduce and study given baseline models. I will invertigate the repo "tree-sitter/tree-sitter" to extract AST info. Besides, I will explore the multi-language issue. If it is not easy to handle six languages, I will start from Java or Python, as discussed with Zimin. For the writting, I will start filling the third chapter (Related Work).

## QUESTIONS
I met some questions in preparing the environment but have solved by meeting with Zimin.

## Special Note
As discussed with Zimin, my experiments will run on Azure VM, not on SNIC. It seems much easier and more clear to prepare the environment on the former than on the latter.