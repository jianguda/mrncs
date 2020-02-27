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

csn_so = 'build/csn.so'
# Language.build_library(
#   csn_so,
#   [
#     'vendor/tree-sitter-go',
#     'vendor/tree-sitter-java',
#     'vendor/tree-sitter-javascript',
#     'vendor/tree-sitter-php',
#     'vendor/tree-sitter-python',
#     'vendor/tree-sitter-ruby',
#   ]
# )

# Load the languages into your app as Language objects:
GO_LANGUAGE = Language(csn_so, 'go')
JAVA_LANGUAGE = Language(csn_so, 'java')
JAVASCRIPT_LANGUAGE = Language(csn_so, 'javascript')
PHP_LANGUAGE = Language(csn_so, 'php')
PYTHON_LANGUAGE = Language(csn_so, 'python')
RUBY_LANGUAGE = Language(csn_so, 'ruby')

# Basic Parsing
# Create a Parser and configure it to use one of the languages:

parser = Parser()
parser.set_language(PYTHON_LANGUAGE)

# Parse some source code:
code = """
def get_vid_from_url(url):
    return match1(url, r'youtu\\.be/([^?/]+)')
     or match1(url, r'youtube\\.com/embed/([^/?]+)')
     or match1(url, r'youtube\\.com/v/([^/?]+)')
     or match1(url, r'youtube\\.com/watch/([^/?]+)')
     or parse_query_param(url, 'v')
     or parse_query_param(parse_query_param(url, 'u'), 'v')
"""
tree = parser.parse(code.encode())

# Inspect the resulting Tree:
root_node = tree.root_node
print(root_node.field_name)
print(root_node.type)
print(root_node.start_point)
print(root_node.end_point)

function_node = root_node.children[0]
print(function_node.type)
print(function_node.child_by_field_name('name').type)

function_name_node = function_node.children[1]
print(function_name_node.type)
print(function_name_node.start_point)
print(function_name_node.end_point)

print(root_node.sexp())

# Walking Syntax Trees
# use TreeCursor to traverse a large number of nodes efficiently
cursor = tree.walk()
print(cursor.node.type)
print(cursor.goto_first_child())
print(cursor.node.type)
print(cursor.goto_first_child())
print(cursor.node.type)

# Returns `False` because the `def` node has no children
print(cursor.goto_first_child())

print(cursor.goto_next_sibling())
print(cursor.node.type)

print(cursor.goto_next_sibling())
print(cursor.node.type)

print(cursor.goto_parent())
print(cursor.node.type)

# /mnt/c/Users/jian/PycharmProjects/csn/code/scripts
