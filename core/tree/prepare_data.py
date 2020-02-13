import itertools
import string
from collections import Counter
from multiprocessing import Pool
from pathlib import Path
from typing import Iterable, List

import numpy as np
from spacy.lang.en.stop_words import STOP_WORDS

from tree import shared, utils
from tree.bpevocabulary import BpeVocabulary
from tree.ts import code2paths, code2sbt, code2lcrs
from tree.desensitizer import desensitize, formalize


def preprocess_tokens(tokens: Iterable[str], data_type: str) -> Iterable[List[str]]:
    for token in tokens:
        if not shared.PROCESSING:
            yield [token]
        # mainly for 'code' and 'query'
        # tokens of 'leaf' have been formalized
        # tokens of 'path', 'sbt', 'lcrs' have been desensitized
        if token.isspace():
            continue
        if token in string.punctuation:
            continue
        if token.isdigit():
            yield ['number']
        if data_type == 'code':
            token = desensitize(token)
            if len(token) <= 1:
                continue
        elif data_type == 'query':
            token = formalize(token)
            if token in STOP_WORDS:
                continue
            if len(token) <= 1:
                continue
        yield token.split('|')


def doc2tokens(doc, language, data_type, evaluation=True):
    if data_type == 'code':
        tokens = doc['function_tokens' if evaluation else 'code_tokens']
    elif data_type == 'rootpath':
        tokens = code2paths(doc['function' if evaluation else 'code'], language, mode='rootpath')
    elif data_type == 'leafpath':
        tokens = code2paths(doc['function' if evaluation else 'code'], language, mode='leafpath')
    elif data_type == 'sbt':
        tokens = code2sbt(doc['function' if evaluation else 'code'], language)
    elif data_type == 'lcrs':
        tokens = code2lcrs(doc['function' if evaluation else 'code'], language)
    else:  # query
        tokens = doc.split() if evaluation else doc['docstring_tokens']
    return tokens


def prepare_corpus_docs(args):
    language, data_set = args
    print(f'Building docs for {language} {data_set}')

    if utils.check_doc(language, data_set):
        return
    prepared_docs = list()
    for doc in utils.get_csn_corpus(language, data_set):
        language = doc['language']
        prepared_doc = dict()
        for data_type in shared.DATA_TYPES:
            tokens = doc2tokens(doc, language, data_type, evaluation=False)
            prepared_doc[data_type] = tokens
        prepared_docs.append(prepared_doc)

    utils.dump_doc(prepared_docs, language, data_set)
    print(f'Done building for {language} {data_set}')


def prepare_corpus_vocabs(args):
    language, data_set, data_type = args
    print(f'Building vocabulary for {language} {data_type}')

    if utils.check_vocab(language, data_type):
        return
    docs = utils.load_doc(language, data_set)
    tokens = (utils.flatten(doc[data_type] for doc in docs))
    tokens = utils.flatten(preprocess_tokens(tokens, data_type))
    vocabulary = BpeVocabulary(vocab_size=shared.VOCAB_SIZE, pct_bpe=shared.VOCAB_PCT_BPE)
    vocabulary.fit(Counter(tokens))
    utils.dump_vocab(vocabulary, language, data_type)

    print(f'Done building vocabulary for {language} {data_type}')


def encode_seqs(seqs: Iterable[List[str]], language: str, data_type: str) -> np.ndarray:
    sentences = (utils.flatten(preprocess_tokens(seq, data_type)) for seq in seqs)
    input_length = utils.get_input_length(data_type)
    vocabs = utils.load_vocab(language, data_type)
    encoded_seqs = vocabs.transform(sentences, fixed_length=input_length)
    return np.array(list(encoded_seqs))


def prepare_seqs(args):
    language, data_set, data_type = args
    print(f'Building sequences for {language} {data_set} {data_type}')

    if utils.check_seq(language, data_set, data_type):
        return
    if data_set == 'evaluation' and data_type == 'query':
        docs = utils.get_csn_queries()
    else:
        docs = utils.load_doc(language, data_set)
    if data_set == 'evaluation':
        seqs = (doc2tokens(doc, language, data_type) for doc in docs)
    else:
        seqs = (doc[data_type] for doc in docs)
    encoded_seqs = encode_seqs(seqs, language, data_type)
    utils.dump_seq(encoded_seqs, language, data_set, data_type)

    print(f'Done building sequences for {language} {data_set} {data_type}')


def prepare_query_docs(args):
    language, data_set = args
    print(f'Parsing {language} {data_set}')
    if utils.check_doc(language, data_set):
        return
    evaluation_pkl_path = Path(shared.DATA_DIR) / f'{language}_dedupe_definitions_v2.pkl'
    evaluation_docs = utils.load_pickle(evaluation_pkl_path)
    utils.dump_doc(evaluation_docs, language, data_set)


def caching():
    print('Caching')
    processors = max(len(shared.LANGUAGES), 4)
    # corpus
    with Pool(processors) as p:
        p.map(prepare_corpus_docs, itertools.product(shared.LANGUAGES, shared.DATA_SETS))
    with Pool(processors) as p:
        p.map(prepare_corpus_vocabs, itertools.product(shared.LANGUAGES, ['train'], shared.DATA_TYPES))
    with Pool(processors) as p:
        p.map(prepare_seqs, itertools.product(shared.LANGUAGES, shared.DATA_SETS, shared.DATA_TYPES))
    # query
    with Pool(processors) as p:
        p.map(prepare_query_docs, itertools.product(shared.LANGUAGES, ['evaluation']))
    with Pool(processors) as p:
        p.map(prepare_seqs, itertools.product(shared.LANGUAGES, ['evaluation'], shared.DATA_TYPES))
