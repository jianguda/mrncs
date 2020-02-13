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

from tree import shared, train_model, utils


class MrrEarlyStopping(EarlyStopping):
    def __init__(self, encoded_seqs_dict: dict):
        super().__init__(monitor='val_mrr', mode='max', restore_best_weights=True, verbose=True, patience=5)
        self.encoded_seqs_dict = encoded_seqs_dict

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}

        mean_mrr = compute_mrr(self.model, self.encoded_seqs_dict)
        print('Mean MRR:', mean_mrr)
        super().on_epoch_end(epoch, {**logs, 'val_mrr': mean_mrr})


def get_embeddings(model, encoded_seqs_dict: dict, idx_chunk):
    predictors = list()
    chunked_seqs_list = list()
    for data_type in shared.SUB_TYPES:
        predictor = train_model.get_embedding_predictor(model, data_type)
        predictors.append(predictor)
        chunked_encoded_seqs = encoded_seqs_dict.get(data_type)[idx_chunk, :]
        if shared.CONTEXT:
            input_ids = chunked_encoded_seqs
            input_masks = np.where(input_ids == chunked_encoded_seqs[data_type], 0, 1)
            input_type_ids = np.zeros_like(input_ids)
            chunked_seqs_list.append([input_ids, input_masks, input_type_ids])
        else:
            chunked_seqs_list.append(chunked_encoded_seqs)
    embeddings_list = list()
    for embedding_predictor, chunked_seqs in zip(predictors, chunked_seqs_list):
        embeddings = embedding_predictor.predict(chunked_seqs)
        embeddings_list.append(embeddings)
    return utils.repack_embeddings(embeddings_list)


def compute_mrr(model, encoded_seqs_dict: dict):
    n_samples = encoded_seqs_dict.get('query').shape[0]
    indices = list(range(n_samples))
    random.shuffle(indices)
    mrr_scores = []
    for idx_chunk in chunked(indices, shared.BATCH_SIZE):
        if len(idx_chunk) < shared.BATCH_SIZE:
            continue
        code_embeddings, query_embeddings = get_embeddings(model, encoded_seqs_dict, idx_chunk)
        distance_matrix = cdist(query_embeddings, code_embeddings, 'cosine')
        correct_elements = np.expand_dims(np.diag(distance_matrix), axis=-1)
        ranks = np.sum(distance_matrix <= correct_elements, axis=-1)
        mrr_scores.append(np.mean(1.0 / ranks))

    return np.mean(mrr_scores)


def emit_mrr_scores(model, language: str):
    valid_seqs_dict = dict()
    test_seqs_dict = dict()
    for data_type in shared.SUB_TYPES:
        valid_seqs_dict[data_type] = utils.load_seq(language, 'valid', data_type)
        test_seqs_dict[data_type] = utils.load_seq(language, 'test', data_type)
    # Check for invalid sequences when it is not for evaluation
    valid_seqs_dict = utils.filter_valid_seqs(valid_seqs_dict)
    test_seqs_dict = utils.filter_valid_seqs(test_seqs_dict)
    valid_mean_mrr = compute_mrr(model, valid_seqs_dict)
    test_mean_mrr = compute_mrr(model, test_seqs_dict)
    return valid_mean_mrr, test_mean_mrr


def emit_ndcg_scores(model, language: str):
    prediction = []
    print(f'Evaluating {language}')

    for data_type in shared.SUB_TYPES:
        # we always rebuild embeddings when it is using attention
        if utils.check_embedding(language, data_type) and not shared.ATTENTION:
            continue
        print(f'Building {data_type} embeddings')
        predictor = train_model.get_embedding_predictor(model, data_type)
        seqs = utils.load_seq(language, 'evaluation', data_type)
        embeddings = predictor.predict(seqs)
        utils.dump_embedding(embeddings, language, data_type)

    print('Loading embeddings')
    embeddings_list = list()
    for data_type in shared.SUB_TYPES:
        embeddings = utils.load_embedding(language, data_type)
        embeddings_list.append(embeddings)

    code_embeddings, query_embeddings = utils.repack_embeddings(embeddings_list)
    evaluation_docs = [{'url': doc['url'], 'identifier': doc['identifier']}
                       for doc in utils.load_doc(language, 'evaluation')]

    print('Indexing embeddings')
    queries = utils.get_csn_queries()
    if shared.ANNOY:
        annoy = AnnoyIndex(shared.EMBEDDING_SIZE, 'angular')
        for idx in range(code_embeddings.shape[0]):
            annoy.add_item(idx, code_embeddings[idx, :])
        annoy.build(10)
        # annoy.build(200)

        for query_idx, query in enumerate(queries):
            query_embedding = query_embeddings[query_idx]
            nearest_indices = annoy.get_nns_by_vector(query_embedding, 100)
            for nearest_idx in nearest_indices:
                prediction.append({
                    'query': query,
                    'language': language,
                    'identifier': evaluation_docs[nearest_idx]['identifier'],
                    'url': evaluation_docs[nearest_idx]['url'],
                })
    else:
        nn = NearestNeighbors(n_neighbors=100, metric='cosine', n_jobs=-1)
        nn.fit(code_embeddings)
        _, nearest_indices = nn.kneighbors(query_embeddings)

        for query_idx, query in enumerate(queries):
            for nearest_idx in nearest_indices[query_idx, :]:
                prediction.append({
                    'query': query,
                    'language': language,
                    'identifier': evaluation_docs[nearest_idx]['identifier'],
                    'url': evaluation_docs[nearest_idx]['url'],
                })

    del evaluation_docs
    gc.collect()
    return prediction


def evaluating():
    print('Evaluating')
    models = dict()
    for language in shared.LANGUAGES:
        model = utils.load_model(language, train_model.get_model())
        models[language] = model

    # emit_mrr_scores
    valid_mrr_scores = {}
    test_mrr_scores = {}
    for language in shared.LANGUAGES:
        model = models.get(language)
        valid_mean_mrr, test_mean_mrr = emit_mrr_scores(model, language)
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
    for language in shared.LANGUAGES:
        model = models.get(language)
        prediction = emit_ndcg_scores(model, language)
        predictions.extend(prediction)

    df_predictions = pd.DataFrame(predictions, columns=['query', 'language', 'identifier', 'url'])
    csv_path = (Path(wandb.run.dir) if shared.WANDB else shared.RESOURCES_DIR) / 'model_predictions.csv'
    df_predictions.to_csv(csv_path, index=False)
