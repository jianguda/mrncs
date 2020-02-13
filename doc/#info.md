\section{SST Transformation Rules}

% In this section, we describe which kind of transformation should be applied to convert AST to SST. 

We have three transformation rules for converting AST to corresponding SST, and concrete operations are listed as follows:

% To make our results more intuitive, we list them as follows.

\paragraph{Prune Semantically Meaningless Tree Nodes}

\begin{compactenum}
\item Remove reserve words, like "self", "this", "super"
\item Remove type declarations, like "int", "boolean", "string"
\item Remove modifier keywords, like "public", "abstract", "abstract"
\item Remove functional keywords, like "async", "await", "lambda"
\item Remove the ExpressionStatement Node
\end{compactenum}

\paragraph{Use Descriptive Tags as Labels of Non-terminal Tree Nodes}

\begin{compactenum}
\item Use "access" for field access
\item Use "assign" for variable declaration
\item Use "invoke" for function invocations
\item Use "number" for exact numbers
\item Use "literal" for exact string variables
\item Use "operate" for exact operators, like "+", "="
\item Use "loop" for for-loop and while-loop statements
\end{compactenum}

\paragraph{Unify the Expression of Semantically Similar Labels from Various Languages}

\begin{compactenum}
\item Unify "block", "statement\_block" as "block"
\item Unify "function", "program", "define", "module" as "module"
\end{compactenum}

\chapter{Experimental Details}
\label{chap:app-details}

\section{Implementation Details}

In this section, we introduce and describe optimizations applied to baseline models after our replication experiments, in case someone is curious about these abrupt changes on results. These optimizations are mainly for the CodeSearchNet Challenge, and meanwhile, they do not affect the conclusions stated in the main text.

Besides, considering the difficulty of exploring new ideas, we turn to our own Keras implementation instead of using the official implementation supplied by the challenge. There might be some minor influences on the concrete value of experimental results, but we have checked that results from two implementations are very close to each other. The most important thing is that our conclusions are consistent.

\input{tables/app-optimization}

As shown in Table \ref{tab:app-optimization}, we introduce four types of optimizations and they are hyper-parameters tuning, exact indexing strategy, data cleaning, and code unifying respectively. In most cases, they bring improvements to both MRR scores and NDCG scores over Python corpus and Ruby Corpus.

\subsection{Model Tuning}

Considering the difficulty of working in the given searching framework, we turn to one relatively simpler Keras implementation. Besides, we make some changes to the hyper-parameters, such as enlarging the embedding size, adjusting the batch size, and switching the optimizer. Meanwhile, we introduce some negative optimizations like removing the Dropout layer to reduce the training time in the expectation of doing more experiments.

\subsection{Indexing Strategy}

When we convert code data and query data to embeddings at the shared vector space, we utilize the nearest neighbor algorithm to index semantically similar code snippets for each natural language query. In the given vanilla implementation, ANNOY, as one approximate nearest neighbor algorithm, is used for indexing. Compared with exact nearest neighbor algorithms, such as KNN, the approximate nearest neighbor is commonly for handling a massive amount of high-dimension data, but with the cost of a certain loss of accuracy. Therefore, the exact nearest neighbor algorithm would be more ideal for the given neural architecture. We turn to use the KNN algorithm to index 1000 nearest code embeddings for each query embedding and find the accuracy scores get significantly improved.

\subsection{Data Cleaning}

In the CodeSearchNet Corpus, raw code snippets contain numerous digitals, literals, and punctuations, but they are merely noise tokens. Besides, for different programming languages, code tokens may have varying levels of semantic quality. For example, Go produces redundant error handling, Java requires explicit type-declaration, and Ruby supports functional programming. The former two may contain lots of valueless tokens and the latter may contain numerous variables named with meaningless identifiers. Based on these ideas, we do data cleaning over code tokens, such as removing punctuations and character tokens, replacing digitals, and literals with corresponding descriptive tags.

For the given evaluation set, some data are like a phrase query. We believe only keywords are needed for the task of semantic code search. There have been some observations \cite{Yan2020AreTC} indicate that a keyword query is more ideal than a phrase query. In contrast, query data in the CodeSearchNet Corpus are usually of low quality. There even exist lots of URLs, HTML tags, and other noise tokens, like JavaDoc keywords. To make sure each query is like a "keyword query", not a "phrase query". We mainly correct irregular writing, and also remove widespread noise tokens, such as punctuations, digitals, stopwords as well as character tokens.

\paragraph{Keyword Query} A keyword query usually contains several keywords that need to be strictly matched with code snippets, like "image encoding base64".

\paragraph{Phrase Query} A phrase query is usually in the form of a sentence or phrase, like "How to convert an image to base64 encoding?".

\subsection{Code Unifying}

To better utilize the multi-language corpus, we implement the idea of unifying code which prompts data of varying programming languages to be more alike. There are three rules which not bring obvious improvements for uni-language learning but are expected to benefit the multi-language learning. The first rule is converting various control-flow statements and identifiers to corresponding descriptive tags, such as converting for-loop, while-loop statements to the "loop" tag, and converting the concrete string to the "literal" tag. The second rule is to remove unnecessary reserve words, such as type declarations like "int" and "boolean", modifier keywords like "public" and "abstract", functional keywords like "async" and "await". The third rule is to unify the expression of various semantically similar tokens, such as unifying "function", "program", "define", "module" as "module", unifying "xor", "not", "in", "===" as "judge". Even though all these processing work could be done automatically or equivalently by a powerful neural network with a massive amount of training data, we do that with the expectation to reduce the requirements for the complexity of models or the quality and quantity of training data.
