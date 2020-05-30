import itertools
import json
import pickle
from pathlib import Path
from typing import Iterable

import numpy as np
import tensorflow as tf

from rok import shared
from rok.bpevocabulary import BpeVocabulary


def repack_embeddings(embeddings_list):
    if len(embeddings_list) == 3:
        hybrid_embeddings = tf.math.add_n(embeddings_list[:2])
        # concat-operation is similar to accumulate-operation
        # concat-operation is to concat distributed embeddings
        # accumulate-operation is to concat one-hot embeddings
        # but concat-operation would double the embedding size
        # therefore we prefer to utilize accumulate-operation
        # hybrid_embeddings = tf.concat(embeddings_list[:2], axis=-1)
        query_embeddings = embeddings_list[2]
        return hybrid_embeddings, query_embeddings
    else:
        return embeddings_list


def get_input_length(data_type: str):
    if data_type == 'code':
        input_length = shared.CODE_MAX_SEQ_LEN
    elif data_type == 'leaf':
        input_length = shared.LEAF_MAX_SEQ_LEN
    elif data_type == 'path':
        input_length = shared.PATH_MAX_SEQ_LEN
    elif data_type == 'sbt':
        input_length = shared.SBT_MAX_SEQ_LEN
    else:  # 'query'
        input_length = shared.QUERY_MAX_SEQ_LEN
    return input_length


def flatten(iterable: Iterable[Iterable[str]]) -> Iterable[str]:
    return itertools.chain.from_iterable(iterable)


def iter_jsonl(file_path: str):
    with open(file_path, encoding='utf-8') as f:
        for line in f:
            yield json.loads(line)


def write_jsonl(iterable, file_path: str):
    with open(file_path, 'w', encoding='utf-8') as f:
        for item in iterable:
            f.write(json.dumps(item) + '\n')


def dump_pickle(obj, serialize_path):
    with open(serialize_path, 'wb') as f:
        pickle.dump(obj, f)


def load_pickle(serialize_path: str):
    with open(serialize_path, 'rb') as f:
        return pickle.load(f)


def get_csn_corpus_path(language: str, data_set: str, idx: int) -> str:
    return Path(shared.DATA_DIR) / language / 'final' / 'jsonl' / data_set / f'{language}_{data_set}_{idx}.jsonl'


def get_csn_corpus(language: str, data_set: str):
    if data_set == 'train':
        file_paths = [get_csn_corpus_path(language, data_set, idx) for idx in range(shared.CORPUS_FILES[language])]
    else:
        file_paths = [get_csn_corpus_path(language, data_set, 0)]

    for file_path in file_paths:
        yield from iter_jsonl(file_path)


def get_csn_queries():
    with open(Path(shared.RESOURCES_DIR) / 'queries.csv', encoding='utf-8') as f:
        return [line.strip() for line in f.readlines()[1:]]


# docs
def _get_docs_path(language: str, data_set: str):
    docs_filename = shared.DOCS_FILENAME.format(language=language, data_set=data_set)
    return Path(shared.DOCS_DIR) / docs_filename


def check_docs(language: str, data_set: str) -> bool:
    path = _get_docs_path(language=language, data_set=data_set)
    return Path(path).exists()


def dump_docs(docs, language: str, data_set: str):
    write_jsonl(docs, _get_docs_path(language=language, data_set=data_set))


def load_docs(language: str, data_set: str):
    return iter_jsonl(_get_docs_path(language=language, data_set=data_set))


# vocabs
def _get_vocabs_path(language: str, data_type: str) -> str:
    vocabs_filename = shared.VOCABS_FILENAME.format(language=language, data_type=data_type)
    return Path(shared.VOCABS_DIR) / vocabs_filename


def check_vocabs(language: str, data_type: str) -> bool:
    path = _get_vocabs_path(language=language, data_type=data_type)
    return Path(path).exists()


def dump_vocabs(vocabulary: BpeVocabulary, language: str, data_type: str):
    dump_pickle(vocabulary, _get_vocabs_path(language=language, data_type=data_type))


def load_vocabs(language: str, data_type: str) -> BpeVocabulary:
    return load_pickle(_get_vocabs_path(language=language, data_type=data_type))


# seqs
def _get_seqs_path(language: str, data_set: str, data_type: str) -> str:
    seqs_filename = shared.SEQS_FILENAME.format(language=language, data_set=data_set, data_type=data_type)
    return Path(shared.SEQS_DIR) / seqs_filename


def check_seqs(language: str, data_set: str, data_type: str) -> bool:
    path = _get_seqs_path(language=language, data_set=data_set, data_type=data_type)
    return Path(path).exists()


def dump_seqs(seqs: np.ndarray, language: str, data_set: str, data_type: str):
    np.save(_get_seqs_path(language=language, data_set=data_set, data_type=data_type), seqs)


def load_seqs(language: str, data_set: str, data_type: str) -> np.ndarray:
    return np.load(_get_seqs_path(language=language, data_set=data_set, data_type=data_type))


# models
def _get_model_path(language: str) -> str:
    models_filename = shared.MODELS_FILENAME.format(language=language)
    return str(Path(shared.MODELS_DIR) / models_filename)


def save_model(language: str, model):
    model.save(_get_model_path(language=language))


def load_model(language: str, model):
    model.load_weights(_get_model_path(language=language), by_name=True)
    return model


# embeddings
def _get_embeddings_path(language: str, data_type: str):
    embeddings_filename = shared.EMBEDDINGS_FILENAME.format(language=language, data_type=data_type)
    return Path(shared.EMBEDDINGS_DIR) / embeddings_filename


def dump_embeddings(code_embeddings: np.ndarray, language: str, data_type: str):
    np.save(_get_embeddings_path(language=language, data_type=data_type), code_embeddings)


def load_embeddings(language: str, data_type: str):
    return np.load(_get_embeddings_path(language=language, data_type=data_type))
