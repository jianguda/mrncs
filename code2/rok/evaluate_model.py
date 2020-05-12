"""
Run predictions on a CodeSearchNet model.

Usage:
    predict.py -m MODEL_FILE [-p PREDICTIONS_CSV]
    predict.py -r RUN_ID     [-p PREDICTIONS_CSV]
    predict.py -h | --help

Options:
    -h --help                       Show this screen
    -m, --model_file FILENAME       Local path to a saved model file (filename.pkl.gz)
    -r, --wandb_run_id RUN_ID       wandb run ID, [username]/codesearchnet/[hash string id], viewable from run overview page via info icon
    -p, --predictions_csv FILENAME  CSV filename for model predictions (note: W&B benchmark submission requires the default name)
                                    [default: ../resources/model_predictions.csv]

Examples:
    ./predict.py -r username/codesearchnet/0123456
    ./predict.py -m ../resources/saved_models/neuralbowmodel-2019-10-31-12-00-00_model_best.pkl.gz
"""

"""
This script tests a model on the CodeSearchNet Challenge, given
- a particular model as a local file (-m, --model_file MODEL_FILENAME.pkl.gz), OR
- as a Weights & Biases run id (-r, --wandb_run_id [username]/codesearchnet/0123456), which you can find
on the /overview page or by clicking the 'info' icon on a given run.
Run with "-h" to see full command line options.
Note that this takes around 2 hours to make predictions on the baseline model.

This script generates ranking results over the CodeSearchNet corpus for a given model by scoring their relevance
(using that model) to 99 search queries of the CodeSearchNet Challenge. We use cosine distance between the learned 
representations of the natural language queries and the code, which is stored in jsonlines files with this format:
https://github.com/github/CodeSearchNet#preprocessed-data-format. The 99 challenge queries are located in 
this file: https://github.com/github/CodeSearchNet/blob/master/resources/queries.csv. 
To download the full CodeSearchNet corpus, see the README at the root of this repository.

Note that this script is specific to methods and code in our baseline model and may not generalize to new models. 
We provide it as a reference and in order to be transparent about our baseline submission to the CodeSearchNet Challenge.

This script produces a CSV file of model predictions with the following fields: 'query', 'language', 'identifier', and 'url':
      * language: the programming language for the given query, e.g. "python".  This information is available as a field in the data to be scored.
      * query: the textual representation of the query, e.g. "int to string" .  
      * identifier: this is an optional field that can help you track your data
      * url: the unique GitHub URL to the returned results, e.g. "https://github.com/JamesClonk/vultr/blob/fed59ad207c9bda0a5dfe4d18de53ccbb3d80c91/cmd/commands.go#L12-L190". This information is available as a field in the data to be scored.

The schema of the output CSV file constitutes a valid submission to the CodeSearchNet Challenge hosted on Weights & Biases. See further background and instructions on the submission process in the root README.

The row order corresponds to the result ranking in the search task. For example, if in row 5 there is an entry for the Python query "read properties file", and in row 60 another result for the Python query "read properties file", then the URL in row 5 is considered to be ranked higher than the URL in row 60 for that query and language.
"""

import gc
import random
import shutil
import sys
from pathlib import Path

from docopt import docopt
import numpy as np
import pandas as pd
import wandb
from wandb.apis import InternalApi
from more_itertools import chunked
from scipy.spatial.distance import cdist
from sklearn.neighbors import NearestNeighbors

from rok import shared, utils, prepare_data
from rok import train_model

np.random.seed(0)
random.seed(0)


def build_code_embeddings():
    for language in shared.LANGUAGES:
        print(f'Building {language} code embeddings')
        model = utils.load_cached_model_weights(language, train_model.get_model())
        code_embedding_predictor = train_model.get_code_embedding_predictor(model)

        evaluation_code_seqs = utils.load_cached_seqs(language, 'evaluation', 'code')
        code_embedding = code_embedding_predictor.predict(evaluation_code_seqs)

        utils.cache_code_embeddings(code_embedding, language)


def emit_ndcg_model_predictions(use_wandb=False):
    build_code_embeddings()
    queries = utils.get_evaluation_queries()

    predictions = []
    for language in shared.LANGUAGES:
        print(f'Evaluating {language}')

        evaluation_docs = [{'url': doc['url'], 'identifier': doc['identifier']}
                           for doc in utils.load_cached_docs(language, 'evaluation')]

        code_embeddings = utils.load_cached_code_embeddings(language)

        model = utils.load_cached_model_weights(language, train_model.get_model())
        query_embedding_predictor = train_model.get_query_embedding_predictor(model)
        query_seqs = prepare_data.pad_encode_seqs(
            prepare_data.preprocess_query_tokens,
            (line.split(' ') for line in queries),
            shared.QUERY_MAX_SEQ_LENGTH,
            language,
            'query')
        query_embeddings = query_embedding_predictor.predict(query_seqs)

        # TODO: Query annoy index
        nn = NearestNeighbors(n_neighbors=100, metric='cosine', n_jobs=-1)
        nn.fit(code_embeddings)
        _, nearest_neighbor_indices = nn.kneighbors(query_embeddings)

        for query_idx, query in enumerate(queries):
            for query_nearest_code_idx in nearest_neighbor_indices[query_idx, :]:
                predictions.append({
                    'query': query,
                    'language': language,
                    'identifier': evaluation_docs[query_nearest_code_idx]['identifier'],
                    'url': evaluation_docs[query_nearest_code_idx]['url'],
                })

        del evaluation_docs
        gc.collect()

    df_predictions = pd.DataFrame(predictions, columns=['query', 'language', 'identifier', 'url'])
    file_name = 'model_predictions.csv'
    save_path = Path(wandb.run.dir) / file_name if use_wandb else f'../resources/{file_name}'
    df_predictions.to_csv(save_path, index=False)


def evaluate_model_mean_mrr(
        model, padded_encoded_code_validation_seqs, padded_encoded_query_validation_seqs, batch_size=1000):
    code_embedding_predictor = train_model.get_code_embedding_predictor(model)
    query_embedding_predictor = train_model.get_query_embedding_predictor(model)

    n_samples = padded_encoded_code_validation_seqs.shape[0]
    indices = list(range(n_samples))
    random.shuffle(indices)
    mrrs = []
    for idx_chunk in chunked(indices, batch_size):
        if len(idx_chunk) < batch_size:
            continue

        code_embeddings = code_embedding_predictor.predict(padded_encoded_code_validation_seqs[idx_chunk, :])
        query_embeddings = query_embedding_predictor.predict(padded_encoded_query_validation_seqs[idx_chunk, :])

        distance_matrix = cdist(query_embeddings, code_embeddings, 'cosine')
        correct_elements = np.expand_dims(np.diag(distance_matrix), axis=-1)
        ranks = np.sum(distance_matrix <= correct_elements, axis=-1)
        mrrs.append(np.mean(1.0 / ranks))

    return np.mean(mrrs)


def evaluate_language_mean_mrr(language):
    model = utils.load_cached_model_weights(language, train_model.get_model())

    valid_code_seqs = utils.load_cached_seqs(language, 'valid', 'code')
    valid_query_seqs = utils.load_cached_seqs(language, 'valid', 'query')
    valid_mean_mrr = evaluate_model_mean_mrr(model, valid_code_seqs, valid_query_seqs)

    test_code_seqs = utils.load_cached_seqs(language, 'test', 'code')
    test_query_seqs = utils.load_cached_seqs(language, 'test', 'query')
    test_mean_mrr = evaluate_model_mean_mrr(model, test_code_seqs, test_query_seqs)

    print(f'Evaluating {language} - Valid Mean MRR: {valid_mean_mrr}, Test Mean MRR: {test_mean_mrr}')
    return valid_mean_mrr, test_mean_mrr


def evaluate_mean_mrr(use_wandb=False):
    language_valid_mrrs = {}
    language_test_mrrs = {}

    for language in shared.LANGUAGES:
        valid_mrr, test_mrr = evaluate_language_mean_mrr(language)
        language_valid_mrrs[f'{language}_valid_mrr'] = valid_mrr
        language_test_mrrs[f'{language}_test_mrr'] = test_mrr

    valid_mean_mrr = np.mean(list(language_valid_mrrs.values()))
    test_mean_mrr = np.mean(list(language_test_mrrs.values()))
    print(f'All languages - Valid Mean MRR: {valid_mean_mrr}, Test Mean MRR: {test_mean_mrr}')

    if use_wandb:
        wandb.log({
            'valid_mean_mrr': valid_mean_mrr,
            'test_mean_mrr': test_mean_mrr,
            **language_valid_mrrs,
            **language_test_mrrs
        })


def submit_to_leaderboard():
    run_id = None
    args_wandb_run_id = args.get('--wandb_run_id')
    predictions_csv = args.get('--predictions_csv')

    if args_wandb_run_id:
        # validate format of runid:
        if len(args_wandb_run_id.split('/')) != 3:
            print("ERROR: Invalid wandb_run_id format: %s (Expecting: user/project/hash)" % args_wandb_run_id, file=sys.stderr)
            sys.exit(1)
        wandb_api = wandb.Api()
        # retrieve saved model from W&B for this run
        print("Fetching run from W&B...")
        try:
            run = wandb_api.run(args_wandb_run_id)
        except wandb.CommError as e:
            print("ERROR: Problem querying W&B for wandb_run_id: %s" % args_wandb_run_id, file=sys.stderr)
            sys.exit(1)

        # print("Fetching run files from W&B...")
        # gz_run_files = [f for f in run.files() if f.name.endswith('gz')]
        # if not gz_run_files:
        #     print("ERROR: Run contains no model-like files")
        #     sys.exit(1)
        # model_file = gz_run_files[0].download(replace=True)
        # local_model_path = model_file.name
        run_id = args_wandb_run_id.split('/')[-1]

    if run_id:
        print('Uploading predictions to W&B')
        # upload model predictions CSV file to W&B

        # we checked that there are three path components above
        entity, project, name = args_wandb_run_id.split('/')

        # make sure the file is in our cwd, with the correct name
        predictions_base_csv = "model_predictions.csv"
        shutil.copyfile(predictions_csv, predictions_base_csv)

        # Using internal wandb API. TODO: Update when available as a public API
        internal_api = InternalApi()
        internal_api.push([predictions_base_csv], run=name, entity=entity, project=project)


if __name__ == '__main__':
    args = docopt(__doc__)
    # evaluate_mean_mrr()
    # emit_ndcg_model_predictions()
    submit_to_leaderboard()
