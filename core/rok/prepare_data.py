import functools
import gc
import itertools
import operator
import re
import string
from collections import Counter
from multiprocessing import Pool
from pathlib import Path
from typing import Iterable, List

import numpy as np
from spacy.lang.en.stop_words import STOP_WORDS

from rok import shared, utils
from rok.bpevocabulary import BpeVocabulary
from rok.ts import code2paths, code2identifiers, code2sbt, code2aux

IDENTIFIER_TOKEN_REGEX = re.compile('[_a-zA-Z][_a-zA-Z0-9]*')
IDENTIFIER_CAMEL_CASE_REGEX = re.compile('.+?(?:(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|$)')


def extract_sub_tokens(token):
    sub_tokens = list()
    for sub_token in re.split('[._]', token):
        sub_tokens.extend(
            IDENTIFIER_CAMEL_CASE_REGEX.findall(sub_token)
            if IDENTIFIER_TOKEN_REGEX.match(sub_token) else [sub_token]
        )

    sub_tokens = [sub_token.lower().strip() for sub_token in sub_tokens]
    return [sub_token for sub_token in sub_tokens if len(sub_token) > 0]


def preprocess_tokens(tokens: Iterable[str], data_type: str) -> Iterable[List[str]]:
    for token in tokens:
        token = token.lower().strip()
        if token in string.punctuation:
            continue
        if data_type == 'query' and token in STOP_WORDS:
            continue
        yield [token]


def doc2tokens(doc, language, data_type, evaluation=True):
    if data_type == 'code':
        tokens = doc['function_tokens' if evaluation else 'code_tokens']
    elif data_type == 'aux':
        tokens = code2aux(doc['function' if evaluation else 'code'], language)
    elif data_type == 'leaf':
        tokens = code2identifiers(doc['function' if evaluation else 'code'], language)
    elif data_type == 'path':
        tokens = code2paths(doc['function' if evaluation else 'code'], language)
    elif data_type == 'sbt':
        tokens = code2sbt(doc['function' if evaluation else 'code'], language)
    else:  # query
        tokens = doc.split() if evaluation else doc['docstring_tokens']
    return tokens


def prepare_corpus_docs(args):
    language, data_set = args
    print(f'Building docs for {language} {data_set}')

    prepared_docs = list()
    for doc in utils.get_csn_corpus(language, data_set):
        language = doc['language']
        prepared_doc = dict()
        for data_type in shared.DATA_TYPES:
            tokens = doc2tokens(doc, language, data_type, evaluation=False)
            prepared_doc.setdefault(data_type, tokens)
        prepared_docs.append(prepared_doc)

    utils.dump_docs(prepared_docs, language, data_set)
    print(f'Done building for {language} {data_set}')


def prepare_vocabs(language):
    print(f'Building vocabulary for {language}')

    docs = utils.load_docs(language, 'train')
    for data_type in shared.SUB_TYPES:
        tokens = (utils.flatten(doc[data_type] for doc in docs))
        tokens = utils.flatten(preprocess_tokens(tokens, data_type))
        vocabulary = BpeVocabulary(vocab_size=shared.VOCAB_SIZE, pct_bpe=shared.VOCAB_PCT_BPE)
        vocabulary.fit(Counter(tokens))
        utils.dump_vocabs(vocabulary, language, data_type)

    print(f'Done building vocabulary for {language}')


def encode_seqs(seqs: Iterable[List[str]], language: str, data_type: str) -> np.ndarray:
    sentences = (utils.flatten(preprocess_tokens(seq, data_type)) for seq in seqs)
    max_length = utils.get_input_length(data_type)
    vocabs = utils.load_vocabs(language, data_type)
    encoded_seqs = vocabs.transform(sentences, fixed_length=max_length)
    return np.array(list(encoded_seqs))


def filter_valid_seqs(encoded_seqs_dict: dict):
    valid_seqs = (encoded_seqs.astype(bool).sum(axis=1) > 0 for encoded_seqs in encoded_seqs_dict.values())
    valid_seqs_indices = functools.reduce(operator.and_, valid_seqs)

    for data_type in encoded_seqs_dict.keys():
        encoded_seqs = encoded_seqs_dict.get(data_type)
        valid_seqs = encoded_seqs[valid_seqs_indices, :]
        encoded_seqs_dict[data_type] = valid_seqs

    return encoded_seqs_dict


def prepare_seqs(args):
    language, data_set = args
    print(f'Building sequences for {language} {data_set}')

    encoded_seqs_dict = dict()
    for data_type in shared.SUB_TYPES:
        if data_set == 'evaluation' and data_type == 'query':
            docs = utils.get_csn_queries()
        else:
            docs = utils.load_docs(language, data_set)
        if data_set == 'evaluation':
            seqs = (doc2tokens(doc, language, data_type) for doc in docs)
        else:
            seqs = (doc[data_type] for doc in docs)
        encoded_seqs = encode_seqs(seqs, language, data_type)
        encoded_seqs_dict.setdefault(data_type, encoded_seqs)
    # Check for invalid sequences
    if data_set != 'evaluation':
        encoded_seqs_dict = filter_valid_seqs(encoded_seqs_dict)
    for data_type, encoded_seqs in encoded_seqs_dict.items():
        utils.dump_seqs(encoded_seqs, language, data_set, data_type)

    print(f'Done building sequences for {language} {data_set}')


def prepare_query_docs(language):
    print(f'Parsing {language}')
    evaluation_pkl_path = Path(shared.DATA_DIR) / f'{language}_dedupe_definitions_v2.pkl'
    evaluation_docs = utils.load_pickle(evaluation_pkl_path)
    utils.dump_docs(evaluation_docs, language, 'evaluation')


def caching():
    # corpus
    with Pool(4) as p:
        p.map(prepare_corpus_docs, itertools.product(shared.LANGUAGES, shared.DATA_SETS))
    with Pool(4) as p:
        p.map(prepare_vocabs, shared.LANGUAGES)
    with Pool(4) as p:
        p.map(prepare_seqs, itertools.product(shared.LANGUAGES, shared.DATA_SETS))
    # query
    with Pool(4) as p:
        p.map(prepare_query_docs, shared.LANGUAGES)
    with Pool(4) as p:
        p.map(prepare_seqs, itertools.product(shared.LANGUAGES, ['evaluation']))
