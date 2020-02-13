from tree_sitter import Language, Parser

# Setup
# First you'll need a Tree-sitter language implementation for
# each language that you want to parse. You can clone some of
# the existing language repos or create your own:
#
# git clone https://github.com/tree-sitter/tree-sitter-go
# git clone https://github.com/tree-sitter/tree-sitter-java
# git clone https://github.com/tree-sitter/tree-sitter-javascript
# git clone https://github.com/tree-sitter/tree-sitter-php
# git clone https://github.com/tree-sitter/tree-sitter-python
# git clone https://github.com/tree-sitter/tree-sitter-ruby

# Use the Language.build_library method to compile these into
# a library that's usable from Python. This function will return
# immediately if the library has already been compiled since the
# last time its source code was modified:

Language.build_library(
  'build/csn-languages.so',
  [
    'vendor/tree-sitter-go',
    'vendor/tree-sitter-java',
    'vendor/tree-sitter-javascript',
    'vendor/tree-sitter-php',
    'vendor/tree-sitter-python',
    'vendor/tree-sitter-ruby',
  ]
)

# Load the languages into your app as Language objects:

GO_LANGUAGE = Language('build/csn-languages.so', 'go')
JAVA_LANGUAGE = Language('build/csn-languages.so', 'java')
JAVASCRIPT_LANGUAGE = Language('build/csn-languages.so', 'javascript')
PHP_LANGUAGE = Language('build/csn-languages.so', 'php')
PYTHON_LANGUAGE = Language('build/csn-languages.so', 'python')
RUBY_LANGUAGE = Language('build/csn-languages.so', 'ruby')

# Basic Parsing
# Create a Parser and configure it to use one of the languages:

parser = Parser()
parser.set_language(PYTHON_LANGUAGE)

print("a")
# Parse some source code:

tree = parser.parse(bytes("""
def foo():
    if bar:
        baz()
""", "utf8"))

# Inspect the resulting Tree:

root_node = tree.root_node
assert root_node.type == 'module'
assert root_node.start_point == (1, 0)
assert root_node.end_point == (3, 13)

function_node = root_node.children[0]
assert function_node.type == 'function_definition'
assert function_node.child_by_field_name('name').type == 'identifier'

function_name_node = function_node.children[1]
assert function_name_node.type == 'identifier'
assert function_name_node.start_point == (1, 4)
assert function_name_node.end_point == (1, 7)

# assert root_node.sexp() == "(module "
#     "(function_definition "
#         "name: (identifier) "
#         "parameters: (parameters) "
#         "body: (block "
#             "(if_statement "
#                 "condition: (identifier) "
#                 "consequence: (block "
#                     "(expression_statement (call "
#                         "function: (identifier) "
#                         "arguments: (argument_list))))))))"

# Walking Syntax Trees
# If you need to traverse a large number of nodes efficiently
# you can use a TreeCursor:

cursor = tree.walk()

assert cursor.node.type == 'module'

assert cursor.goto_first_child()
assert cursor.node.type == 'function_definition'

assert cursor.goto_first_child()
assert cursor.node.type == 'def'

# Returns `False` because the `def` node has no children
assert not cursor.goto_first_child()

assert cursor.goto_next_sibling()
assert cursor.node.type == 'identifier'

assert cursor.goto_next_sibling()
assert cursor.node.type == 'parameters'

assert cursor.goto_parent()
assert cursor.node.type == 'function_definition'

# Editing
# When a source file is edited, you can edit the syntax tree
# to keep it in sync with the source:

tree.edit(
    start_byte=5,
    old_end_byte=5,
    new_end_byte=5 + 2,
    start_point=(0, 5),
    old_end_point=(0, 5),
    new_end_point=(0, 5 + 2),
)

# Then, when you're ready to incorporate the changes into a
# new syntax tree, you can call Parser.parse again, but pass
# in the old tree:

# new_tree = parser.parse(new_source, tree)

# This will run much faster than if you were parsing from scratch.


# Pattern-matching
# You can search for patterns in a syntax tree using a tree query:

query = PYTHON_LANGUAGE.query("""
(function_definition
  name: (identifier) @function.def)

(call
  function: (identifier) @function.call)
""")

captures = query.captures(tree.root_node)
assert len(captures) == 2
assert captures[0][0] == function_name_node
assert captures[0][1] == "function.def"

# csn
# https://github.com/github/CodeSearchNet/blob/master/function_parser/script/setup.py
