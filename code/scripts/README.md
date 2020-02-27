## Parser
The Python parser is used to produced [Python150k](https://eth-sri.github.io/py150)-style AST files. In order to support more programming languages, we plan to use tree-sitter later.

The [tree-sitter](https://tree-sitter.github.io/tree-sitter/) parser is used because it is modern and universal, also it is used for CodeSearchNet corpus. The work is dependent on its Python bindings. Because of some potential issues, it can not work smoothly on `Windows`, so `Ubuntu` is recommended.

The parser is to extract ASTs from source code.

## Extractor
The extractor is to generate Paths from ASTs

## Preprocess
It do some preprocess work for code2seq.
