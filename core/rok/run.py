import os
import sys
from time import strftime
from pathlib import Path

import wandb

from rok import prepare_data, train_model, evaluate_model, shared


def main():
    commands = sys.argv[1:]

    do_caching = False
    for path_name in (
            shared.CACHES_DIR, shared.DOCS_DIR, shared.VOCABS_DIR,
            shared.SEQS_DIR, shared.MODELS_DIR, shared.EMBEDDINGS_DIR
    ):
        if not os.path.exists(Path(path_name)):
            do_caching = True
            os.makedirs(Path(path_name))
    if 'fully' in commands or do_caching:
        prepare_data.caching()

    if shared.WANDB:
        project = 'CodeSearchNet'
        name = 'kth-%s-%s' % (shared.MODE_TAG, strftime('%Y-%m-%d'))
        wandb.init(project=project, name=name)

    for language in shared.LANGUAGES:
        print(f'Training {language}')
        train_model.training(language)

    print('Evaluating')
    evaluate_model.evaluating()


# JGD todo
#  the accuracy is much lower than expected
if __name__ == '__main__':
    main()
