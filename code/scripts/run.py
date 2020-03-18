import pickle
import traceback
from collections import Counter
from glob import glob
from pathlib import Path

from dpu_utils.utils import load_jsonl_gz

from scripts.extract import tree2paths
from scripts.parse import code2tree
from scripts.ts import TS


# using TreeSitter, for all 6 languages
def code2paths(code):
    ts = TS(code)
    paths = ts.code2paths()
    return paths


# using Python's AST module, only for Python
def code2paths4py(code):
    tree = code2tree(code)
    paths = tree2paths(tree)
    return paths


def contexts2tokens(contexts):
    paths = list()
    for context in contexts:
        paths.extend(context)
    paths = [path.split(',') for path in paths]
    terminals = list()
    items = map(lambda x: x[0] + '|' + x[2], paths)
    for item in items:
        terminals.extend(item.split('|'))
    nonterminals = list()
    items = map(lambda x: x[1], paths)
    for item in items:
        nonterminals.extend(item.split('|'))
    return terminals, nonterminals


def collect_filenames(path):
    pattern = path / "**/*.jsonl.gz"
    filenames = glob(str(pattern), recursive=True)
    return filenames


def prepare_data(filenames):
    data = list()
    for filename in filenames:
        for sample in load_jsonl_gz(filename):
            code = sample['code']
            data.append(code)
    return data


def print_data(ast_paths, terminals_stats, nonterminals_stats):
    print(f'{"@" * 9}ast_paths\n{ast_paths}')
    print(f'{"@" * 9}terminals_stats\n{terminals_stats}')
    print(f'{"@" * 9}nonterminals_stats\n{nonterminals_stats}')


def load_data(path, lang):
    contexts_file = path / f'{lang}_contexts.csv'
    ast_contexts = list()
    with open(contexts_file, 'r') as file:
        context_lines = file.readlines()
        for context_line in context_lines:
            ast_paths = context_line.split()
            ast_contexts.append(ast_paths)
        print(f'ast_contexts loaded from: {contexts_file}')
    stats_file = path / f'{lang}_stats.pkl'
    with open(stats_file, 'rb') as file:
        terminals_stats = pickle.load(file)
        nonterminals_stats = pickle.load(file)
        print(f'stats data loaded from: {stats_file}')
    return ast_paths, terminals_stats, nonterminals_stats


def dump_data(path, lang, ast_paths, terminals_stats, nonterminals_stats):
    contexts_file = path / f'{lang}_contexts.csv'
    with open(contexts_file, 'w') as file:
        file.write(' '.join(ast_paths) + '\n')
        print(f'ast_contexts saved to: {contexts_file}')
    stats_file = path / f'{lang}_stats.pkl'
    with open(stats_file, 'wb') as file:
        pickle.dump(terminals_stats, file)
        pickle.dump(nonterminals_stats, file)
        print(f'stats data saved to: {stats_file}')


def process_data(data):
    success_num = 0
    error_num = 0
    ast_contexts = list()
    for code in data:
        try:
            # ast_paths = code2paths(code)
            ast_paths = code2paths4py(code)
            ast_contexts.append(ast_paths)
            success_num += 1
        except (AttributeError, IndexError, SyntaxError, TypeError):
            error_num += 1
            traceback.print_exc()
    print(f'success_num:{success_num}\nerror_num:{error_num}')
    terminals, nonterminals = contexts2tokens(ast_contexts)
    # count the tokens
    terminals_stats = list(Counter(terminals).items())
    nonterminals_stats = list(Counter(nonterminals).items())
    return ast_contexts, terminals_stats, nonterminals_stats


def run4corpus():
    lang = 'python'
    path = Path(f'C:\\Users\\jian\\Documents\\Corpus\\{lang}\\{lang}\\final\\jsonl\\train')
    filenames = collect_filenames(path)
    data = prepare_data(filenames)
    ast_contexts, terminals_stats, nonterminals_stats = process_data(data)
    dump_data(path, lang, ast_contexts, terminals_stats, nonterminals_stats)


def run4file():
    filenames = ['python_test_0.jsonl.gz']
    data = prepare_data(filenames)
    ast_paths, terminals_stats, nonterminals_stats = process_data(data)
    print_data(ast_paths, terminals_stats, nonterminals_stats)


def run4demo():
    code = """def get_vid_from_url(url):
    return match1(url, r'youtube\\.com/v/([^/?]+)')\
        or parse_query_param(url, 'v')\
        or parse_query_param(parse_query_param(url, 'u'), 'v')"""
    data = [code]
    ast_paths, terminals_stats, nonterminals_stats = process_data(data)
    print_data(ast_paths, terminals_stats, nonterminals_stats)


if __name__ == '__main__':
    # todo
    #  1 work on the brew-data part
    #  2 work on the data-loader part
    #  3 refactor the alon encoder
    #  ......
    #  6 regenerate preprocessed data
    #  7 run experiments and debug
    #  8 improve the tree-sitter parser
    #  9 make func-name as identifier as well
    # run4corpus()
    # run4file()
    run4demo()
