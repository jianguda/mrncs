import gc
import itertools
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
from rok.ts import code2identifiers

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


def prepare_corpus_docs(args):
    language, data_set = args
    print(f'Building docs for {language} {data_set}')

    docs = list()
    for corpus_doc in utils.get_csn_corpus(language, data_set):
        query_tokens = corpus_doc['docstring_tokens']
        if shared.SIAMESE:
            code_tokens = code2identifiers(corpus_doc['code'], corpus_doc['language'])
        else:
            code_tokens = corpus_doc['code_tokens']

        docs.append({
            'code_tokens': code_tokens,
            'query_tokens': query_tokens,
            'url': corpus_doc['url'],
        })

    utils.dump_docs(docs, language, data_set)
    print(f'Done building for {language} {data_set}')


def prepare_corpus_vocabs(args):
    language, data_type = args
    print(f'Building vocabulary for {language} {data_type}')

    docs = utils.load_docs(language, 'train')
    tokens = (utils.flatten(doc[f'{data_type}_tokens'] for doc in docs))
    tokens = utils.flatten(preprocess_tokens(tokens, data_type))
    vocabulary = BpeVocabulary(vocab_size=shared.VOCAB_SIZE, pct_bpe=shared.VOCAB_PCT_BPE)
    vocabulary.fit(Counter(tokens))
    utils.dump_vocabs(vocabulary, language, data_type)

    print(f'Done building vocabulary for {language} {data_type}')


def encode_seqs(seqs: Iterable[List[str]], language: str, data_type: str) -> np.ndarray:
    sentences = (utils.flatten(preprocess_tokens(seq, data_type)) for seq in seqs)
    max_length = shared.CODE_MAX_SEQ_LEN if data_type == 'code' else shared.QUERY_MAX_SEQ_LEN
    vocabs = utils.load_vocabs(language, data_type)
    encoded_seqs = vocabs.transform(sentences, fixed_length=max_length)
    return np.array(list(encoded_seqs))


def keep_valid_seqs(padded_encoded_code_seqs, padded_encoded_query_seqs):
    # Keep seqs with at least one valid token
    valid_code_seqs = padded_encoded_code_seqs.astype(bool).sum(axis=1) > 0
    valid_query_seqs = padded_encoded_query_seqs.astype(bool).sum(axis=1) > 0
    valid_seqs_indices = valid_code_seqs & valid_query_seqs

    return padded_encoded_code_seqs[valid_seqs_indices, :], padded_encoded_query_seqs[valid_seqs_indices, :]


def prepare_corpus_seqs(args):
    language, data_set = args
    print(f'Building sequences for {language} {data_set}')

    # Prepare code seqs
    code_seqs = (doc['code_tokens'] for doc in utils.load_docs(language, data_set))
    encoded_code_seqs = encode_seqs(code_seqs, language, 'code')
    # Prepare query seqs
    query_seqs = (doc['query_tokens'] for doc in utils.load_docs(language, data_set))
    encoded_query_seqs = encode_seqs(query_seqs, language, 'query')
    # Check for invalid sequences
    encoded_code_seqs, encoded_query_seqs = keep_valid_seqs(encoded_code_seqs, encoded_query_seqs)

    utils.dump_seqs(encoded_code_seqs, language, data_set, 'code')
    utils.dump_seqs(encoded_query_seqs, language, data_set, 'query')

    print(f'Done building sequences for {language} {data_set}')


def prepare_evaluation_seqs(language):
    print(f'Building evaluation sequences for {language}')

    evaluation_docs = utils.load_docs(language, 'evaluation')
    if shared.SIAMESE:
        evaluation_code_seqs = (code2identifiers(doc['function'], language) for doc in evaluation_docs)
    else:
        evaluation_code_seqs = (doc['function_tokens'] for doc in evaluation_docs)

    encoded_code_seqs = encode_seqs(evaluation_code_seqs, language, 'code')
    utils.dump_seqs(encoded_code_seqs, language, 'evaluation', 'code')

    del evaluation_code_seqs
    gc.collect()

    queries = utils.get_csn_queries()
    evaluation_query_seqs = (line.split(' ') for line in queries)
    encoded_query_seqs = encode_seqs(evaluation_query_seqs, language, 'query')
    utils.dump_seqs(encoded_query_seqs, language, 'evaluation', 'query')

    print(f'Done building evaluation sequences for {language}')


def prepare_evaluation_docs(language):
    print(f'Parsing {language}')
    evaluation_docs_pkl_path = Path(shared.DATA_DIR) / f'{language}_dedupe_definitions_v2.pkl'
    evaluation_docs = utils.load_pickle(evaluation_docs_pkl_path)
    utils.dump_docs(evaluation_docs, language, 'evaluation')


def caching():
    # corpus
    with Pool(4) as p:
        p.map(prepare_corpus_docs, itertools.product(shared.LANGUAGES, shared.DATA_SETS))
    with Pool(4) as p:
        p.map(prepare_corpus_vocabs, itertools.product(shared.LANGUAGES, shared.DATA_TYPES))
    with Pool(4) as p:
        p.map(prepare_corpus_seqs, itertools.product(shared.LANGUAGES, shared.DATA_SETS))
    # evaluation
    with Pool(2) as p:
        p.map(prepare_evaluation_docs, shared.LANGUAGES)
    with Pool(2) as p:
        p.map(prepare_evaluation_seqs, shared.LANGUAGES)
