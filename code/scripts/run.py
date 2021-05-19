import pickle
import traceback
from collections import Counter
from glob import glob
from pathlib import Path

from dpu_utils.utils import load_jsonl_gz

from .ts import code2identifiers, code2paths, code2paths4py, code2sexp


# JGD for alon_encoder
def paths2tokens(paths):
    paths = [path.split(',') for path in paths]
    terminals = list()
    nonterminals = []
    items = map(lambda x: x[0] + '|' + x[2], paths)
    for item in items:
        terminals.extend(item.split('|'))
    items = map(lambda x: x[1], paths)
    for item in items:
        nonterminals.extend(item.split('|'))
    return terminals, nonterminals


def collect_filenames(path):
    pattern = path / "**/*_0.jsonl.gz"
    # pattern = path / "**/*.jsonl.gz"
    filenames = glob(str(pattern), recursive=True)
    return filenames


def prepare_data(filenames, key='code'):
    for filename in filenames:
        for sample in load_jsonl_gz(filename):
            value = sample[key]
            yield value


def print_data(terminal_counter, nonterminal_counter):
    print(f'{"@" * 9}terminal_counter\n{terminal_counter}')
    print(f'{"@" * 9}nonterminal_counter\n{nonterminal_counter}')


# JGD for alon_encoder
def get_path(language, locally=False):
    if locally:
        return Path(f'/home/jian/data/{language}/final/jsonl/train')
            # path = Path(f'C:\\Users\\jian\\Documents\\Corpus\\{language}\\final\\jsonl\\train')
    else:
        return Path(f'/home/dev/resources/data/{language}/final/jsonl/train')
            # path = Path(f'/datadrive/CodeSearchNet/resources/data/{language}/final/jsonl/train')


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


def process_data(data, language='python', path=None):
    success_num = 0
    error_num = 0
    terminal_counter = Counter()
    nonterminal_counter = Counter()
    for code in data:
        try:
            # identifiers = code2identifiers(code, language)
            # print(identifiers)
            # tree_paths = code2paths(code, language)
            # print(tree_paths)
            # JGD consider the top1M paths, just like in code2vec
            tree_paths = code2paths(code, language)
            # tree_paths = code2paths4py(code)
            terminals, nonterminals = paths2tokens(tree_paths)
            terminal_counter += Counter(terminals)
            nonterminal_counter += Counter(nonterminals)
            if path is None:
                print('\n'.join(tree_paths))
            else:
                contexts_file = path / f'contexts.csv'
                with open(contexts_file, 'a') as file:
                    file.write('\n'.join(tree_paths) + '\n')
                    # print(f'contexts saved to: {contexts_file}')
            success_num += 1
            print(f'success_num:{success_num}\t\terror_num:{error_num}', end='\r')
        except (AttributeError, IndexError, SyntaxError, TypeError):
            error_num += 1
            traceback.print_exc()
    return terminal_counter, nonterminal_counter
    # return ast_contexts, None, None


def check_data(language='python', key='docstring_tokens'):
    path = get_path(language, True)
    print('A')
    filenames = collect_filenames(path)
    print('B')
    print(filenames)
    data = prepare_data(filenames, key)
    print('C')
    for datum in data:
        print(datum)


def run4corpus(language='python', locally=False):
    path = get_path(language, locally)
    filenames = collect_filenames(path)
    data = prepare_data(filenames)
    terminal_counter, nonterminal_counter = process_data(data, language, path)
    dump_data(terminal_counter, nonterminal_counter, path)


def run4file(language='python'):
    path = Path()
    filenames = ['python_test_0.jsonl.gz']
    data = prepare_data(filenames)
    terminal_counter, nonterminal_counter = process_data(data, language, path)
    dump_data(terminal_counter, nonterminal_counter, path)
    # print_data(terminal_counter, nonterminal_counter)


def run4demo():
    go_code = """func (s *SkuM1Small) GetInnkeeperClient() (innkeeperclient.InnkeeperClient, error) {\n\tvar err error\n\tif s.Client == nil {\n\t\tif clnt, err := s.InitInnkeeperClient(); err == nil {\n\t\t\ts.Client = clnt\n\t\t} else {\n\t\t\tlo.G.Error(\"error parsing current cfenv: \", err.Error())\n\t\t}\n\t}\n\treturn s.Client, err\n}"""
    java_code = """protected void notifyAttemptToReconnectIn(int seconds) {\n        if (isReconnectionAllowed()) {\n            for (ConnectionListener listener : connection.connectionListeners) {\n                listener.reconnectingIn(seconds);\n            }\n        }\n    }"""
    javascript_code = """function (context, grunt) {\n  this.context = context;\n  this.grunt = grunt;\n\n  // Merge task-specific and/or target-specific options with these defaults.\n  this.options = context.options(defaultOptions);\n}"""
    php_code = """public function init()\n    {\n        parent::init();\n        if ($this->message === null) {\n            $this->message = \\Reaction::t('rct', '{attribute} is invalid.');\n        }\n    }"""
    python_code = """def get_url_args(url):\n    \"\"\" Returns a dictionary from a URL params \"\"\"\n    url_data = urllib.parse.urlparse(url)\n    arg_dict = urllib.parse.parse_qs(url_data.query)\n    return arg_dict"""
    ruby_code = """def part(name)\n      parts.select {|p| p.name.downcase == name.to_s.downcase }.first\n    end"""

    data = [go_code, java_code, javascript_code, php_code, python_code, ruby_code]
    languages = ['go', 'java', 'javascript', 'php', 'python', 'ruby']
    # process_data([go_code], 'go')
    # process_data([java_code], 'java')
    # process_data([javascript_code], 'javascript')
    # process_data([php_code], 'php')
    # process_data([python_code], 'python')
    # process_data([ruby_code], 'ruby')
    for code, language in zip(data, languages):
        code2sexp(code, language)
        # terminal_counter, nonterminal_counter = process_data([code], language)
        # print_data(terminal_counter, nonterminal_counter)
