import pickle
import sys
import traceback
from collections import Counter
from glob import glob
from pathlib import Path

from dpu_utils.utils import load_jsonl_gz

from .extract import tree2paths
from .parse import code2tree
from .ts import PathExtractor


# using TreeSitter, for all 6 languages
def code2paths(code):
    ts = PathExtractor(code)
    paths = ts.code2paths()
    return paths


# using Python's AST module, only for Python
def code2paths4py(code):
    tree = code2tree(code)
    paths = tree2paths(tree)
    return paths


# JGD for alon_encoder
def paths2tokens(paths):
    paths = [path.split(',') for path in paths]
    terminals = list()
    nonterminals = list()
    items = map(lambda x: x[0] + '|' + x[2], paths)
    for item in items:
        terminals.extend(item.split('|'))
    items = map(lambda x: x[1], paths)
    for item in items:
        nonterminals.extend(item.split('|'))
    return terminals, nonterminals


def collect_filenames(path):
    # pattern = path / "**/*_0.jsonl.gz"
    pattern = path / "**/*.jsonl.gz"
    filenames = glob(str(pattern), recursive=True)
    return filenames


def prepare_data(filenames):
    for filename in filenames:
        for sample in load_jsonl_gz(filename):
            code = sample['code']
            yield code


def print_data(ast_contexts, terminal_counter, nonterminal_counter):
    print(f'{"@" * 9}ast_contexts\n{ast_contexts}')
    print(f'{"@" * 9}terminal_counter\n{terminal_counter}')
    print(f'{"@" * 9}nonterminal_counter\n{nonterminal_counter}')


# JGD for alon_encoder
def get_path(language, locally=False):
    if locally:
        path = Path(f'/home/jian/data/{language}/final/jsonl/train')
        # path = Path(f'C:\\Users\\jian\\Documents\\Corpus\\{language}\\final\\jsonl\\train')
    else:
        path = Path(f'/home/dev/resources/data/{language}/final/jsonl/train')
        # path = Path(f'/datadrive/CodeSearchNet/resources/data/{language}/final/jsonl/train')
    return path


# JGD for alon_encoder
def load_data(path):
    contexts_file = path / f'contexts.csv'
    # ast_contexts = list()
    # with open(contexts_file, 'r') as file:
    #     context_lines = file.readlines()
    #     for context_line in context_lines:
    #         ast_paths = context_line.split()
    #         ast_contexts.append(ast_paths)
    #     print(f'contexts loaded from: {contexts_file}')
    context_filename = str(contexts_file)
    counters_file = path / f'counters.pkl'
    with open(counters_file, 'rb') as file:
        terminal_counter = pickle.load(file)
        nonterminal_counter = pickle.load(file)
        # print(f'counters loaded from: {counters_file}')
    # return ast_contexts, terminal_counter, nonterminal_counter
    return context_filename, terminal_counter, nonterminal_counter


def dump_data(terminal_counter, nonterminal_counter, path):
    counters_file = path / f'counters.pkl'
    with open(counters_file, 'wb') as file:
        pickle.dump(terminal_counter, file)
        pickle.dump(nonterminal_counter, file)
        # print(f'counters saved to: {counters_file}')


def process_data(data, path=None):
    success_num = 0
    error_num = 0
    terminal_counter = Counter()
    nonterminal_counter = Counter()
    for code in data:
        try:
            ast_paths = code2paths(code)
            # ast_paths = code2paths4py(code)
            terminals, nonterminals = paths2tokens(ast_paths)
            terminal_counter += Counter(terminals)
            nonterminal_counter += Counter(nonterminals)
            if path is not None:
                contexts_file = path / f'contexts.csv'
                with open(contexts_file, 'a') as file:
                    file.write('\n'.join(ast_paths) + '\n')
                    # print(f'contexts saved to: {contexts_file}')
            success_num += 1
            print(f'success_num:{success_num}\t\terror_num:{error_num}', end='\r')
        except (AttributeError, IndexError, SyntaxError, TypeError):
            error_num += 1
            traceback.print_exc()
            # sys.exit()
    return terminal_counter, nonterminal_counter
    # return ast_contexts, None, None


def run4corpus(language='python', locally=False):
    path = get_path(language, locally)
    filenames = collect_filenames(path)
    data = prepare_data(filenames)
    terminal_counter, nonterminal_counter = process_data(data, path)
    dump_data(terminal_counter, nonterminal_counter, path)


def run4file():
    path = Path()
    filenames = ['python_test_0.jsonl.gz']
    data = prepare_data(filenames)
    terminal_counter, nonterminal_counter = process_data(data, path)
    dump_data(terminal_counter, nonterminal_counter, path)
    # print_data(ast_contexts, terminal_counter, nonterminal_counter)


def run4demo():
    code = """def sina_xml_to_url_list(xml_data):
    rawurl = []
    dom = parseString(xml_data)
    for node in dom.getElementsByTagName('durl'):
        url = node.getElementsByTagName('url')[0]
        rawurl.append(url.childNodes[0].data)
    return rawurl"""
    data = [code]
    ast_contexts, terminal_counter, nonterminal_counter = process_data(data)
    print_data(ast_contexts, terminal_counter, nonterminal_counter)
