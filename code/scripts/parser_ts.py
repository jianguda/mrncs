import re
import itertools
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

# Parse some source code:
code = """
def get_vid_from_url(url):
    return match1(url, r'youtube\\.com/v/([^/?]+)')
     + parse_query_param(url, 'v')
     + parse_query_param(parse_query_param(url, 'u'), 'v')
"""


def demo(code):
    parser = Parser()
    parser.set_language(PYTHON_LANGUAGE)
    # Inspect the resulting Tree:
    tree = parser.parse(code.encode())
    root = tree.root_node
    print(root.type)
    function_node = root.children[0]
    print(function_node.type)

    print(root.sexp())

    # terminals = gen_terminals(root)
    # for terminal in terminals:
    #     print('*' * 16)
    #     print(terminal)

    # paths = gen_paths(root)
    # for path in paths:
    #     print('*' * 16)
    #     print(path)

    tree_paths = gen_paths(root)
    contexts = []
    for tree_path in tree_paths:
        start, connector, finish = tree_path

        start, finish = tokenize(start), tokenize(finish)
        connector = '|'.join(connector)

        context = f'{start},{connector},{finish}'
        contexts.append(context)

    if len(contexts) == 0:
        return None

    function_name_node = function_node.child_by_field_name('name')
    target = node2value(function_name_node)
    target = tokenize(target)
    context = ' '.join(contexts)

    print(f'{target} {context}'.replace(' ', '\n'))
    return f'{target} {context}'


def tokenize(name):
    def camel_case_split(identifier):
        matches = re.finditer(
            '.+?(?:(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|$)',
            identifier,
        )
        return [m.group(0) for m in matches]

    blocks = []
    for underscore_block in name.split('_'):
        blocks.extend(camel_case_split(underscore_block))

    return '|'.join(block.lower() for block in blocks)


def node2value(node):
    line_start = node.start_point[0]
    line_end = node.end_point[0]
    char_start = node.start_point[1]
    char_end = node.end_point[1]

    if line_start != line_end:
        return None

    lines = code.split('\n')
    return lines[line_start][char_start:char_end]


def lookup(node):
    node_type, node_value = None, None
    if node.type == 'call':
        node_type = 'Call'
        node_value = 'Call'
    elif node.type == 'unary_operator':
        node_type = 'unary_operator'
        node = node.children[0]
        node_value = node2value(node)
    elif node.type == 'identifier':
        node_type = 'identifier'
        node_value = node2value(node)
        if node_value == 'or':
            node_type = 'BoolOr'
            node_value = 'BoolOr'
    return node_type, node_value


def gen_terminals(root):
    path, paths = [], []

    def dfs(node):
        flag = False
        node_type, node_value = lookup(node)
        if node_value is not None:
            if node_type == 'identifier':
                paths.append((path.copy(), node_value))
            else:
                flag = True
                path.append(node_value)
        if node.type == 'function_definition':
            for child in node.children:
                if child.type == 'block':
                    dfs(child)
        else:
            for child in node.children:
                dfs(child)
        if flag:
            path.pop()

    dfs(root)
    return paths


def leaves2paths(v_path, u_path):
    s, n, m = 0, len(v_path), len(u_path)
    while s < min(n, m) and v_path[s] == u_path[s]:
        s += 1

    prefix = list(reversed(v_path[s:]))
    lca = v_path[s - 1]
    suffix = u_path[s:]

    return prefix, lca, suffix


def gen_paths(root):
    terminals = gen_terminals(root)

    tree_paths = []
    max_path_length = 8
    max_path_width = 2
    for (v_path, v_value), (u_path, u_value) in itertools.combinations(iterable=terminals, r=2):
        prefix, lca, suffix = leaves2paths(v_path, u_path)
        if (len(prefix) + 1 + len(suffix) <= max_path_length) \
                and (abs(len(prefix) - len(suffix)) <= max_path_width):
            path = prefix + [lca] + suffix
            tree_path = v_value, path, u_value
            tree_paths.append(tree_path)

    return tree_paths


# cd /mnt/c/Users/jian/PycharmProjects/csn/code/scripts
demo(code)

# todo test on more samples
# todo check on other languages
