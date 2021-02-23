import random
import re
from itertools import combinations, chain
from queue import Queue

from tree import utils
from tree_sitter import Language, Parser

from tree.desensitizer import desensitize, formalize


class TSNode:
    def __init__(self):
        self.type = None
        self.value = None
        # distinguish with each other even when other info are similar
        self.mark = '@'
        # for the form of LC-RS tree
        self.guardian = None
        self.left_child = None
        self.right_sibling = None
        # for the form of multi-way tree
        self.parent = None
        self.children = list()

    def gen_root_path(self, tree_style='SPT'):
        ptr = self
        root_path = []
        hpt_ptr = None
        while ptr.parent:
            ptr = ptr.parent
            # AST | SPT || HST | HPT
            if tree_style == 'AST':
                # abstract syntax tree
                root_path.append(ptr.type + self.mark)
            elif tree_style == 'SPT':
                # simplified parse tree
                root_path.append(ptr.value + self.mark)
            elif tree_style == 'HST':
                # binary syntax tree
                # hierarchy syntax tree
                root_path.append(ptr.type + self.mark)
                if not hpt_ptr and ptr.type == 'expression_statement':
                    hpt_ptr = ptr
                    root_path = []
            elif tree_style == 'HPT':
                # binary parse tree
                # hierarchy parse tree
                root_path.append(ptr.value + self.mark)
                if not hpt_ptr and ptr.type == 'expression_statement':
                    hpt_ptr = ptr
                    root_path = []
        root_path = list(reversed(root_path))
        value = self.value
        if hpt_ptr:
            values = list()
            q = Queue()
            q.put(hpt_ptr)
            while not q.empty():
                ptr = q.get()
                if ptr.type == 'identifier':
                    values.extend(ptr.value.split('|'))
                for child in ptr.children:
                    q.put(child)
            value = '|'.join(values)
        return root_path, value + self.mark


class TS:
    def __init__(self, code, language='python', tree_style='SPT', path_style='L2L'):
        # AST | SPT || HST | HPT
        self.tree_style = tree_style
        # L2L | UD | U2D
        self.path_style = path_style
        # Use the Language.build_library method to compile these
        # into a library that's usable from Python:
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
        parser = Parser()
        # Load the languages into your app as Language objects:
        # ('go', 'java', 'javascript', 'php', 'python', 'ruby')
        parser.set_language(Language(csn_so, language))
        tree = parser.parse(code.encode())
        code_lines = code.split('\n')
        self.root, self.terminals = self.traverse(tree, code_lines)
        self.nonterminals = list()
        self.leafpath_terminals = list()
        self.leafpath_nonterminals = list()
        self.rootpath_terminals = list()
        self.rootpath_nonterminals = list()
        self.debug = False
        if self.debug:
            print(f'{"@" * 9}code\n{code}')
            print(f'{"@" * 9}sexp\n{tree.root_node.sexp()}')

    def traverse(self, tree, code_lines):
        q = Queue()
        terminals = list()
        root = TSNode()
        q.put((root, tree.root_node))
        while not q.empty():
            # lhs is the node we defined
            # rhs is the node TS supplied
            lhs, rhs = q.get()
            lhs.type = str(rhs.type).lower().strip()
            lhs.value = self.query_token(rhs, code_lines)
            # mark is "@1~4~1~7" if start_point == (1, 4) and end_point == (1, 7)
            lhs.mark += '~'.join(rhs.start_point + rhs.end_point)
            if rhs.children:
                # non-terminals
                lhs.value = desensitize(lhs.value)
                left_sibling = None
                for rhs_child in rhs.children:
                    lhs_child = TSNode()
                    # for the form of LC-RS tree
                    if left_sibling:
                        lhs_child.guardian = left_sibling
                        left_sibling.right_sibling = lhs_child
                    else:
                        lhs_child.guardian = lhs
                        lhs.left_child = lhs_child
                    left_sibling = lhs_child
                    # for the form of multi-way tree
                    lhs_child.parent = lhs
                    lhs.children.append(lhs_child)
                    q.put((lhs_child, rhs_child))
            else:
                # terminals
                lhs.value = self.tokenize(lhs.value)
                lhs.value = formalize(lhs.value)
                terminals.append(lhs)

        return root, terminals

    def gen_root_paths(self):
        threshold = 20
        root_paths_1 = list()  # number of qualified leaf node 1
        root_paths_0 = list()  # number of qualified leaf node 0
        for terminal in self.terminals:
            root_path, value = terminal.gen_root_path(self.tree_style)
            self.nonterminals.extend(root_path)
            if terminal.type == 'identifier':
                if root_path and value:
                    if len(value) > 1:
                        root_paths_1.append((root_path, value))
                    else:
                        root_paths_0.append((root_path, value))
        root_paths = root_paths_1
        margin = threshold - len(root_paths)
        if margin < 0:
            root_paths = random.sample(root_paths, threshold)
        elif margin > 0:
            margin = min(margin, len(root_paths_0))
            root_paths_0 = random.sample(root_paths_0, margin)
            root_paths.extend(root_paths_0)
        if self.debug:
            print(f'{"@" * 9}root_paths\n{root_paths}')
        for root_path, value in root_paths:
            self.rootpath_terminals.append(value)
            self.rootpath_nonterminals.extend(root_path)
        return root_paths

    def gen_leaf_paths(self):
        threshold = 20
        path_width_threshold = 2
        path_length_threshold = 8
        root_paths = self.gen_root_paths()
        cases = combinations(iterable=root_paths, r=2)
        leaf_paths_2 = list()  # number of qualified leaf node 2
        leaf_paths_1 = list()  # number of qualified leaf node 1
        leaf_paths_0 = list()  # number of qualified leaf node 0
        for (u_path, u_value), (v_path, v_value) in cases:
            prefix, lca, suffix = self.merge_paths(u_path, v_path)
            prefix_len = len(prefix)
            suffix_len = len(suffix)
            if threshold <= len(leaf_paths_2):
                if len(u_value) <= 1 or len(v_value) <= 1:
                    continue
            elif threshold <= len(leaf_paths_2) + len(leaf_paths_1):
                if len(u_value) <= 1 and len(v_value) <= 1:
                    continue
            if 1 <= prefix_len and 1 <= suffix_len \
                    and abs(prefix_len - suffix_len) <= path_width_threshold \
                    and prefix_len + 1 + suffix_len <= path_length_threshold:
                source, target = u_value, v_value
                if self.path_style == 'L2L':
                    middle = '|'.join(prefix + [lca] + suffix)
                elif self.path_style == 'UD':
                    middle = '|'.join('U' * prefix_len + 'D' * suffix_len)
                else:
                    middle = '|U|'.join(prefix) + f'|U|{lca}|D|' + '|D|'.join(suffix)
                # leaf_path = middle
                leaf_path = f'{source}|{middle}|{target}'
                if len(source) > 1 or len(target) > 1:
                    if len(source) > 1 and len(target) > 1:
                        leaf_paths_2.append(leaf_path)
                    else:
                        leaf_paths_1.append(leaf_path)
                else:
                    leaf_paths_0.append(leaf_path)
        leaf_paths = leaf_paths_2
        margin = threshold - len(leaf_paths)
        if margin < 0:
            leaf_paths = random.sample(leaf_paths, threshold)
        elif margin > 0:
            margin = min(margin, len(leaf_paths_1))
            leaf_paths_1 = random.sample(leaf_paths_1, margin)
            leaf_paths.extend(leaf_paths_1)
            margin = threshold - len(leaf_paths)
            if margin > 0:
                margin = min(margin, len(leaf_paths_0))
                leaf_paths_0 = random.sample(leaf_paths_0, margin)
                leaf_paths.extend(leaf_paths_0)
        if self.debug:
            print(f'{"@" * 9}leaf_paths\n{leaf_paths}')
        for leaf_path in leaf_paths:
            source, middle, target = leaf_path.split('|')
            self.leafpath_terminals.append(source)
            self.leafpath_terminals.append(target)
            self.leafpath_nonterminals.extend(middle)
        return leaf_paths

    def stats(self):
        # deduplication
        self.nonterminals = list(set(self.nonterminals))
        self.rootpath_nonterminals = list(set(self.rootpath_nonterminals))
        self.leafpath_nonterminals = list(set(self.leafpath_nonterminals))
        # path_coverage = len(path_nonterminals) / len(nonterminals)
        num_nonterminals = len(self.nonterminals)
        num_rootpath_nonterminals = len(self.rootpath_nonterminals)
        num_leafpath_nonterminals = len(self.leafpath_nonterminals)
        path_coverage_rootpath = num_rootpath_nonterminals / num_nonterminals
        path_coverage_leafpath = num_leafpath_nonterminals / num_nonterminals

        # deduplication
        self.terminals = list(set(self.clean_mark(self.terminals)))
        self.nonterminals = list(set(self.clean_mark(self.nonterminals)))
        self.rootpath_terminals = list(set(self.clean_mark(self.rootpath_terminals)))
        self.rootpath_nonterminals = list(set(self.clean_mark(self.rootpath_nonterminals)))
        self.leafpath_terminals = list(set(self.clean_mark(self.leafpath_terminals)))
        self.leafpath_nonterminals = list(set(self.clean_mark(self.leafpath_nonterminals)))
        # node_coverage = len(path_nonterminals + path_terminals) / len(cleaned_nonterminals + cleaned_terminals)
        num_terminals = len(self.terminals)
        num_nonterminals = len(self.nonterminals)
        num_rootpath_terminals = len(self.rootpath_terminals)
        num_rootpath_nonterminals = len(self.rootpath_nonterminals)
        num_leafpath_terminals = len(self.leafpath_terminals)
        num_leafpath_nonterminals = len(self.leafpath_nonterminals)
        node_coverage_rootpath = (num_rootpath_terminals + num_rootpath_nonterminals) / (num_terminals + num_nonterminals)
        node_coverage_leafpath = (num_leafpath_terminals + num_leafpath_nonterminals) / (num_terminals + num_nonterminals)

        return path_coverage_rootpath, path_coverage_leafpath, node_coverage_rootpath, node_coverage_leafpath

    @staticmethod
    def clean_mark(nodes):
        nodes = [node.split('@')[0] for node in nodes]
        return nodes

    @staticmethod
    def query_token(node, code_lines):
        line_start = node.start_point[0]
        line_end = node.end_point[0]
        char_start = node.start_point[1]
        char_end = node.end_point[1]

        if line_start != line_end:
            token = code_lines[line_start][char_start:]
        else:
            token = code_lines[line_start][char_start:char_end]
        return token

    @staticmethod
    def tokenize(term):
        def camel_case_split(identifier):
            matches = re.finditer(
                '.+?(?:(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|$)',
                identifier,
            )
            return [m.group(0) for m in matches]

        blocks = []
        for underscore_block in term.split('_'):
            blocks.extend(camel_case_split(underscore_block))
        return '|'.join(block.lower() for block in blocks)

    @staticmethod
    def merge_paths(u_path, v_path):
        m, n, s = len(u_path), len(v_path), 0
        while s < min(m, n) and u_path[s] == v_path[s]:
            s += 1

        prefix = list(reversed(u_path[s:]))
        lca = u_path[s - 1]
        suffix = v_path[s:]

        return prefix, lca, suffix


def code2paths(code, language='python', mode='rootpath'):
    ts = TS(code, language)
    paths = list()
    if mode == 'rootpath':
        root_paths = ts.gen_root_paths()
        for (root_path, identifier) in root_paths:
            paths.extend(root_path)
            paths.extend(identifier.split('|'))
    else:
        leaf_paths = ts.gen_leaf_paths()
        for leaf_path in leaf_paths:
            paths.extend(leaf_path.split('|'))
    return paths


def doc2tokens(doc, language, data_type, evaluation=True):
    if data_type == 'code':
        tokens = doc['function_tokens' if evaluation else 'code_tokens']
    elif data_type == 'rootpath':
        tokens = code2paths(doc['function' if evaluation else 'code'], language, mode='rootpath')
    elif data_type == 'leafpath':
        tokens = code2paths(doc['function' if evaluation else 'code'], language, mode='leafpath')
    return tokens


def run_stats(language):
    sum_path_coverage_rootpath = 0.0
    sum_path_coverage_leafpath = 0.0
    sum_node_coverage_rootpath = 0.0
    sum_node_coverage_leafpath = 0.0
    train_docs = utils.get_csn_corpus(language, 'train')
    valid_docs = utils.get_csn_corpus(language, 'valid')
    test_docs = utils.get_csn_corpus(language, 'test')
    docs_counter = 0
    for doc in chain(train_docs, valid_docs, test_docs):
        language = doc['language']
        code = doc['code']
        ts = TS(code, language)
        # we only invoke ts.gen_leaf_paths()
        # because ts.gen_root_paths() will be invoked there
        _ = ts.gen_leaf_paths()
        # stats
        path_coverage_rootpath, path_coverage_leafpath, node_coverage_rootpath, node_coverage_leafpath = ts.stats()
        sum_path_coverage_rootpath += path_coverage_rootpath
        sum_path_coverage_leafpath += path_coverage_leafpath
        sum_node_coverage_rootpath += node_coverage_rootpath
        sum_node_coverage_leafpath += node_coverage_leafpath
        docs_counter += 1
    print(f'language={language}')
    avg_path_coverage_rootpath = sum_path_coverage_rootpath / docs_counter
    avg_path_coverage_leafpath = sum_path_coverage_leafpath / docs_counter
    avg_node_coverage_rootpath = sum_node_coverage_rootpath / docs_counter
    avg_node_coverage_leafpath = sum_node_coverage_leafpath / docs_counter
    print(f'avg_path_coverage_rootpath={avg_path_coverage_rootpath}')
    print(f'avg_path_coverage_leafpath={avg_path_coverage_leafpath}')
    print(f'avg_node_coverage_rootpath={avg_node_coverage_rootpath}')
    print(f'avg_node_coverage_leafpath={avg_node_coverage_leafpath}')



if __name__ == '__main__':
    run_stats('ruby')
    run_stats('python')
