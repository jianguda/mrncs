import functools
import operator
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
    if len(embeddings_list) >= 3:
        # concat-operation is similar to accumulate-operation
        # concat-operation is to concat distributed embeddings
        # accumulate-operation is to concat one-hot embeddings
        # but concat-operation would double the embedding size
        # therefore we prefer to utilize accumulate-operation
        hybrid_embeddings = tf.math.add_n(embeddings_list[:-1])
        # hybrid_embeddings = tf.concat(embeddings_list[:2], axis=-1)
        # Hadamard product (element-wise) is not the better choice!
        # hybrid_embeddings = tf.math.multiply(embeddings_list[0], embeddings_list[1])
        query_embeddings = embeddings_list[-1]
        return hybrid_embeddings, query_embeddings
    else:
        return embeddings_list


def get_input_length(data_type: str):
    if data_type == 'query':
        input_length = shared.QUERY_SEQ_LEN
    else:  # 'code', 'rootpath', 'leafpath', 'sbt', 'lcrs'
        input_length = shared.CODE_SEQ_LEN
    return input_length


def get_compatible_mode_tag():
    if shared.ANNOY:
        mode_tag = 'annoy'
    elif shared.ATTENTION:
        mode_tag = 'attention'
    elif shared.DESENSITIZE:
        mode_tag = 'desensitize'
    else:
        mode_tag = shared.MODE_TAG
    return mode_tag


def filter_valid_seqs(encoded_seqs_dict: dict):
    valid_seqs = (encoded_seqs.astype(bool).sum(axis=1) > 0 for encoded_seqs in encoded_seqs_dict.values())
    valid_seqs_indices = functools.reduce(operator.and_, valid_seqs)

    for data_type in encoded_seqs_dict.keys():
        encoded_seqs = encoded_seqs_dict.get(data_type)
        valid_seqs = encoded_seqs[valid_seqs_indices, :]
        encoded_seqs_dict[data_type] = valid_seqs

    return encoded_seqs_dict


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
        file_paths = [get_csn_corpus_path(language, data_set, idx)
                      for idx in range(shared.CORPUS_FILES[language])]
    else:
        file_paths = [get_csn_corpus_path(language, data_set, 0)]

    for file_path in file_paths:
        yield from iter_jsonl(file_path)


def get_csn_queries():
    with open(Path(shared.RESOURCES_DIR) / 'queries.csv', encoding='utf-8') as f:
        return [line.strip() for line in f.readlines()[1:]]


# docs
def _get_doc_path(language: str, data_set: str):
    filename = shared.DOC_FILENAME.format(language=language, data_set=data_set)
    return Path(shared.DOCS_DIR) / filename


def check_doc(language: str, data_set: str) -> bool:
    path = _get_doc_path(language=language, data_set=data_set)
    return Path(path).exists()


def dump_doc(docs, language: str, data_set: str):
    write_jsonl(docs, _get_doc_path(language=language, data_set=data_set))


def load_doc(language: str, data_set: str):
    return iter_jsonl(_get_doc_path(language=language, data_set=data_set))


# vocabs
def _get_vocab_path(language: str, data_type: str) -> str:
    filename = shared.VOCAB_FILENAME.format(language=language, data_type=data_type)
    return Path(shared.VOCABS_DIR) / filename


def check_vocab(language: str, data_type: str) -> bool:
    path = _get_vocab_path(language=language, data_type=data_type)
    return Path(path).exists()


def dump_vocab(vocabulary: BpeVocabulary, language: str, data_type: str):
    dump_pickle(vocabulary, _get_vocab_path(language=language, data_type=data_type))


def load_vocab(language: str, data_type: str) -> BpeVocabulary:
    return load_pickle(_get_vocab_path(language=language, data_type=data_type))


# seqs
def _get_seq_path(language: str, data_set: str, data_type: str) -> str:
    filename = shared.SEQ_FILENAME.format(language=language, data_set=data_set, data_type=data_type)
    return Path(shared.SEQS_DIR) / filename


def check_seq(language: str, data_set: str, data_type: str) -> bool:
    path = _get_seq_path(language=language, data_set=data_set, data_type=data_type)
    return Path(path).exists()


def dump_seq(seqs: np.ndarray, language: str, data_set: str, data_type: str):
    np.save(_get_seq_path(language=language, data_set=data_set, data_type=data_type), seqs)


def load_seq(language: str, data_set: str, data_type: str) -> np.ndarray:
    return np.load(_get_seq_path(language=language, data_set=data_set, data_type=data_type))


# models
def _get_model_path(language: str) -> str:
    mode_tag = get_compatible_mode_tag()
    filename = shared.MODEL_FILENAME.format(language=language, mode_tag=mode_tag)
    return str(Path(shared.MODELS_DIR) / filename)


def check_model(language: str) -> bool:
    path = _get_model_path(language=language)
    return Path(path).exists() and shared.LAZY


def save_model(language: str, model):
    model.save(_get_model_path(language=language))


def load_model(language: str, model):
    model.load_weights(_get_model_path(language=language), by_name=True)
    return model


# embeddings
def _get_embedding_path(language: str, data_type: str):
    mode_tag = get_compatible_mode_tag()
    filename = shared.EMBEDDING_FILENAME.format(
        language=language, data_type=data_type, mode_tag=mode_tag)
    return Path(shared.EMBEDDINGS_DIR) / filename


def check_embedding(language: str, data_type: str) -> bool:
    path = _get_embedding_path(language=language, data_type=data_type)
    return Path(path).exists() and shared.LAZY


def dump_embedding(code_embeddings: np.ndarray, language: str, data_type: str):
    np.save(_get_embedding_path(language=language, data_type=data_type), code_embeddings)


def load_embedding(language: str, data_type: str):
    return np.load(_get_embedding_path(language=language, data_type=data_type))
