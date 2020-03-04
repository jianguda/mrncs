import traceback
from glob import glob
from pathlib import Path

from dpu_utils.utils import load_jsonl_gz

from code.scripts.extract import collect_samples
from code.scripts.parser import parse_file
from code.scripts.preprocess import build_items, dump_items


def process(path, role):
    code_items = list()
    pattern = path / role / "**/*.jsonl.gz"
    file_names = glob(str(pattern), recursive=True)

    num_error = 0
    for file_name in file_names:
        curr = 0
        for sample in load_jsonl_gz(file_name):
            print(f'{file_name}:{curr}')
            curr += 1
            try:
                # load data
                code = sample['code']
                # parse AST
                code_ast = parse_file(code)
                # extract AST-path
                code_item = collect_samples(code_ast)[0]
                code_items.append(code_item)
            except (AttributeError, IndexError, SyntaxError, TypeError):
                num_error += 1
                traceback.print_exc()
    print(num_error)

    context_items = list()
    label_items = list()
    leaf_items = list()
    node_items = list()
    curr = 0
    for code_item in code_items:
        print(f'#{curr}')
        curr += 1
        # process AST-path
        context_item, label_item, leaf_item, node_item = build_items(code_item)
        context_items.extend(context_item)
        label_items.extend(label_item)
        leaf_items.extend(leaf_item)
        node_items.extend(node_item)

    dump_items(path, role, context_items, label_items, leaf_items, node_items)


def demo():
    code = """
def get_vid_from_url(url):
    return match1(url, r'youtube\\.com/v/([^/?]+)')\
        or parse_query_param(url, 'v')\
        or parse_query_param(parse_query_param(url, 'u'), 'v')
        """
    code_ast = parse_file(code)
    print(code_ast)
    code_paths = collect_samples(code_ast)[0]
    print(code_paths.replace(' ', '\n'))
    # context_item, label_item, leaf_item, node_item = build_items(code_paths)
    # print(context_item)
    # print(label_item)
    # print(leaf_item)
    # print(node_item)


def main():
    path = Path('C:\\Users\\jian\\Documents\\Corpus\\python\\python\\final\\jsonl')
    # process(path, 'train')
    process(path, 'test')
    # process(path, 'valid')


if __name__ == '__main__':
    # todo switch from py150k parser to tree-sitter parser
    # main()
    demo()
