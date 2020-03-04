import pickle
import random
from collections import Counter


def build_items(code_item, max_contexts=200):
    labels, *contexts = code_item.strip().split()
    label_item = labels.split('|')

    context_lines = ' '.join(contexts).split()
    context_lines = [context_line.split(',') for context_line in context_lines]
    leaf_item = list()
    tmp_lines = map(lambda x: x[0] + '|' + x[2], context_lines)
    for tmp_line in tmp_lines:
        leaf_item.extend(tmp_line.split('|'))
    node_item = list()
    tmp_lines = map(lambda x: x[1], context_lines)
    for tmp_line in tmp_lines:
        node_item.extend(tmp_line.split('|'))

    if len(contexts) > max_contexts:
        # truncates methods with many contexts
        random.seed(999)
        contexts = random.sample(contexts, max_contexts)
    else:
        # pads methods having less paths with spaces
        contexts.extend([''] * (max_contexts - len(contexts)))
    context_item = [labels] + contexts

    return context_item, label_item, leaf_item, node_item


def dump_items(path, role, context_items, label_items, leaf_items, node_items):
    context_path = path / f'{role}.context.c2s'
    with open(context_path, 'wb') as file:
        pickle.dump(context_items, file)
        print(f'context data saved to: {context_path}')

    # count the labels in the function name
    label_stats = list(Counter(label_items).items())
    # count the tokens in the leafs
    leaf_stats = list(Counter(leaf_items).items())
    # count the tokens in the nodes
    node_stats = list(Counter(node_items).items())

    stats_path = path / f'{role}.stats.c2s'
    with open(stats_path, 'wb') as file:
        pickle.dump(label_stats, file)
        pickle.dump(leaf_stats, file)
        pickle.dump(node_stats, file)
        print(f'stats data saved to: {stats_path}')
