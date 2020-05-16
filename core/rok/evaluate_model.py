import gc
import random
from pathlib import Path

import numpy as np
import pandas as pd
import wandb
from annoy import AnnoyIndex
from more_itertools import chunked
from scipy.spatial.distance import cdist
from sklearn.neighbors import NearestNeighbors

from rok import shared, utils, prepare_data
from rok import train_model

np.random.seed(0)
random.seed(0)


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

        annoy_index_flag = True
        if annoy_index_flag:
            annoy = AnnoyIndex(shared.EMBEDDING_SIZE, 'angular')

            for idx in range(code_embeddings.shape[0]):
                annoy.add_item(idx, code_embeddings[idx, :])
            annoy.build(10)

            for query_idx, query in enumerate(queries):
                query_embedding = query_embeddings[query_idx]
                nearest_neighbor_indices = annoy.get_nns_by_vector(query_embedding, 100)
                for query_nearest_code_idx in nearest_neighbor_indices:
                    predictions.append({
                        'query': query,
                        'language': language,
                        'identifier': evaluation_docs[query_nearest_code_idx]['identifier'],
                        'url': evaluation_docs[query_nearest_code_idx]['url'],
                    })
        else:
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


if __name__ == '__main__':
    evaluate_mean_mrr()
    emit_ndcg_model_predictions()
