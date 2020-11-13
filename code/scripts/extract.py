import re
import itertools

METHOD_NAME, NUM = 'METHOD_NAME', 'NUM'


def terminals(ast, node_index):
    stack, paths = [], []

    def dfs(v):
        stack.append(v)

        v_node = ast[v]

        if 'value' in v_node:
            if v == node_index:  # Top-level func def node.
                paths.append((stack.copy(), METHOD_NAME))
            else:
                v_type = v_node['type']

                if v_type.startswith('Name'):
                    paths.append((stack.copy(), v_node['value']))
                elif v_type == 'Num':
                    paths.append((stack.copy(), NUM))
                else:
                    pass

        if 'children' in v_node:
            for child in v_node['children']:
                dfs(child)

        stack.pop()

    dfs(node_index)

    return paths


def merge_terminals2_paths(v_path, u_path):
    s, n, m = 0, len(v_path), len(u_path)
    while s < min(n, m) and v_path[s] == u_path[s]:
        s += 1

    prefix = list(reversed(v_path[s:]))
    lca = v_path[s - 1]
    suffix = u_path[s:]

    return prefix, lca, suffix


def raw_tree_paths(ast, node_index):
    tnodes = terminals(ast, node_index)
    print(f"tnodes\n{tnodes}")
    tree_paths = []
    max_path_length = 8
    max_path_width = 2
    for (v_path, v_value), (u_path, u_value) in itertools.combinations(
            iterable=tnodes,
            r=2,
    ):
        prefix, lca, suffix = merge_terminals2_paths(v_path, u_path)
        if (len(prefix) + 1 + len(suffix) <= max_path_length) \
                and (abs(len(prefix) - len(suffix)) <= max_path_width):
            path = prefix + [lca] + suffix
            tree_path = v_value, path, u_value
            tree_paths.append(tree_path)

    return tree_paths


def delim_name(name):
    if name in {METHOD_NAME, NUM}:
        return name

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


def collect_sample(ast, fd_index):
    root = ast[fd_index]
    if root['type'] != 'FunctionDef':
        raise ValueError('Wrong node type.')

    tree_paths = raw_tree_paths(ast, fd_index)
    contexts = []
    for tree_path in tree_paths:
        start, connector, finish = tree_path

        start, finish = delim_name(start), delim_name(finish)
        connector = '|'.join(ast[v]['type'] for v in connector)

        context = f'{start},{connector},{finish}'
        contexts.append(context)

    if len(contexts) == 0:
        return None

    return ' '.join(contexts)


def tree2paths(tree):
    samples = []
    for node_index, node in enumerate(tree):
        if node['type'] == 'FunctionDef':
            sample = collect_sample(tree, node_index)
            if sample is not None:
                samples.append(sample)

    ast_paths = samples[0].strip().split()
    return ast_paths
