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
from tensorflow.keras.callbacks import EarlyStopping

from rok import shared, train_model, utils


class MrrEarlyStopping(EarlyStopping):
    def __init__(self, code_seqs, query_seqs):
        super().__init__(monitor='val_mrr', mode='max', restore_best_weights=True, verbose=True, patience=5)
        self.code_seqs = code_seqs
        self.query_seqs = query_seqs

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}

        mean_mrr = compute_mrr(self.model, self.code_seqs, self.query_seqs)
        print('Mean MRR:', mean_mrr)
        super().on_epoch_end(epoch, {**logs, 'val_mrr': mean_mrr})


def compute_mrr(model, code_seqs, query_seqs):
    code_embedding_predictor = train_model.get_embedding_predictor(model, 'code')
    query_embedding_predictor = train_model.get_embedding_predictor(model, 'query')

    n_samples = code_seqs.shape[0]
    indices = list(range(n_samples))
    random.shuffle(indices)
    mrr_scores = []
    for idx_chunk in chunked(indices, shared.BATCH_SIZE):
        if len(idx_chunk) < shared.BATCH_SIZE:
            continue

        code_embeddings = code_embedding_predictor.predict(code_seqs[idx_chunk, :])
        query_embeddings = query_embedding_predictor.predict(query_seqs[idx_chunk, :])

        distance_matrix = cdist(query_embeddings, code_embeddings, 'cosine')
        correct_elements = np.expand_dims(np.diag(distance_matrix), axis=-1)
        ranks = np.sum(distance_matrix <= correct_elements, axis=-1)
        mrr_scores.append(np.mean(1.0 / ranks))

    return np.mean(mrr_scores)


def emit_mrr_scores(language: str):
    model = utils.load_model(language, train_model.get_model())

    valid_code_seqs = utils.load_seqs(language, 'valid', 'code')
    valid_query_seqs = utils.load_seqs(language, 'valid', 'query')
    valid_mean_mrr = compute_mrr(model, valid_code_seqs, valid_query_seqs)
    test_code_seqs = utils.load_seqs(language, 'test', 'code')
    test_query_seqs = utils.load_seqs(language, 'test', 'query')
    test_mean_mrr = compute_mrr(model, test_code_seqs, test_query_seqs)
    return valid_mean_mrr, test_mean_mrr


def build_embeddings(language: str, data_type: str):
    print(f'building {language} {data_type} embeddings')
    model = utils.load_model(language, train_model.get_model())
    embedding_predictor = train_model.get_embedding_predictor(model, data_type)

    seqs = utils.load_seqs(language, 'evaluation', data_type)
    embeddings = embedding_predictor.predict(seqs)
    utils.dump_embeddings(embeddings, language, data_type)


def emit_ndcg_scores(language: str, queries):
    prediction = []
    print(f'Evaluating {language}')
    build_embeddings(language, 'code')
    build_embeddings(language, 'query')

    code_embeddings = utils.load_embeddings(language, 'code')
    query_embeddings = utils.load_embeddings(language, 'query')

    evaluation_docs = [{'url': doc['url'], 'identifier': doc['identifier']}
                       for doc in utils.load_docs(language, 'evaluation')]

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
                prediction.append({
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
                prediction.append({
                    'query': query,
                    'language': language,
                    'identifier': evaluation_docs[query_nearest_code_idx]['identifier'],
                    'url': evaluation_docs[query_nearest_code_idx]['url'],
                })

    del evaluation_docs
    gc.collect()
    return prediction


def evaluating():
    # emit_mrr_scores
    valid_mrr_scores = {}
    test_mrr_scores = {}

    for language in shared.LANGUAGES:
        valid_mean_mrr, test_mean_mrr = emit_mrr_scores(language)
        print(f'{language} - Valid Mean MRR: {valid_mean_mrr}, Test Mean MRR: {test_mean_mrr}')
        valid_mrr_scores[f'{language}_valid_mrr'] = valid_mean_mrr
        test_mrr_scores[f'{language}_test_mrr'] = test_mean_mrr

    valid_mean_mrr = np.mean(list(valid_mrr_scores.values()))
    test_mean_mrr = np.mean(list(test_mrr_scores.values()))
    print(f'All languages - Valid Mean MRR: {valid_mean_mrr}, Test Mean MRR: {test_mean_mrr}')

    if shared.WANDB:
        wandb.log({
            'valid_mean_mrr': valid_mean_mrr,
            'test_mean_mrr': test_mean_mrr,
            **valid_mrr_scores,
            **test_mrr_scores
        })

    # emit_ndcg_scores
    predictions = []
    queries = utils.get_csn_queries()
    for language in shared.LANGUAGES:
        prediction = emit_ndcg_scores(language, queries)
        predictions.extend(prediction)

    df_predictions = pd.DataFrame(predictions, columns=['query', 'language', 'identifier', 'url'])
    csv_path = (Path(wandb.run.dir) if shared.WANDB else shared.RESOURCES_DIR) / 'model_predictions.csv'
    df_predictions.to_csv(csv_path, index=False)
