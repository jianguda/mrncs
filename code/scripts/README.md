## parse.py
The Python parser is used to produced [Python150k](https://eth-sri.github.io/py150)-style AST files. In order to support more programming languages, I plan to use tree-sitter later

The [tree-sitter](https://tree-sitter.github.io/tree-sitter/) parser is used because it is modern and universal, also it is used for CodeSearchNet corpus. The work is dependent on its Python bindings. Because of some potential issues, it can not work smoothly on `Windows`, so `Ubuntu` is recommended

The parser is to extract ASTs from source code

## extract.py
The extractor is to generate AST-Paths from ASTs

## ts.py
The parser and extractor in the tree-sitter plan

## run.py
it is used to check the implementation of parsers and extractors. It can be used to generate preprocessed data beforehand