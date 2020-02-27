import json
from glob import glob
from pathlib import Path

from dpu_utils.utils import load_jsonl_gz, save_jsonl_gz
from code.scripts.parser import parse_file
from code.scripts.extract import collect_samples


def collect_files(dir_name):
    pattern = Path(dir_name) / "**/*.jsonl.gz"
    # pattern = Path(dir_name) / "**/_*.jsonl.gz"
    # print(pattern)
    for filename in glob(str(pattern), recursive=True):
        # print(filename)
        handle_data(filename)


def handle_data(filename):
    for sample in load_jsonl_gz(filename):
        # load data
        code = sample['code']
        print(code)
        # parse AST

        # extract Path

        # pre-process

        # save data
        path = Path(filename)
        sample['ast'] = None
        save_jsonl_gz(sample, '_' + path.name)


def pipeline():
    code = """
def get_vid_from_url(url):
    return match1(url, r'youtu\\.be/([^?/]+)')\
        or match1(url, r'youtube\\.com/embed/([^/?]+)')\
        or match1(url, r'youtube\\.com/v/([^/?]+)')\
        or match1(url, r'youtube\\.com/watch/([^/?]+)')\
        or parse_query_param(url, 'v')\
        or parse_query_param(parse_query_param(url, 'u'), 'v')
        """
    code_ast = parse_file(code)
    print(code_ast)
    code_ast = json.loads(code_ast.strip())
    code_paths = collect_samples(code_ast)
    print(code_paths)


if __name__ == '__main__':
    # collect_files('')
    pipeline()
