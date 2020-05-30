import sys
from time import strftime
from pathlib import Path

import wandb

from rok import prepare_data, train_model, evaluate_model, shared


def main():
    commands = sys.argv[1:]

    for path_name in (
        shared.DOCS_DIR, shared.VOCABS_DIR, shared.SEQS_DIR,
        shared.MODELS_DIR, shared.EMBEDDINGS_DIR
    ):
        if not Path(path_name).exists():
            commands.append('cache')
            Path(path_name).mkdir(parents=True)

    if 'fully' in commands:
        commands.extend(['cache', 'train', 'evaluate'])
    if 'cache' in commands:
        prepare_data.caching()
    if shared.WANDB:
        project = 'CodeSearchNet'
        name = 'kth-%s-%s' % (shared.MODE_TAG, strftime('%Y-%m-%d'))
        wandb.init(project=project, name=name)
    if 'train' in commands:
        # JGD (multi-lang)
        for language in shared.LANGUAGES:
            print(f'Training {language}')
            train_model.training(language)
    if 'evaluate' in commands:
        print('Evaluating')
        evaluate_model.evaluating()


# JGD todo
#  check self-attention
#  improve multi-lang
if __name__ == '__main__':
    main()
