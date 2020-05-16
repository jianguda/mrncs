from pathlib import Path
from typing import Generator, Iterable

SIAMESE_MODEL = True

TokensGenerator = Generator[Iterable[str], None, None]

ROOT_DIR = Path.cwd().parent
MODELS_CACHE_DIR = ROOT_DIR / 'cache' / 'models'
SEQS_CACHE_DIR = ROOT_DIR / 'cache' / 'seqs'
VOCABULARIES_CACHE_DIR = ROOT_DIR / 'cache' / 'vocabularies'
DOCS_CACHE_DIR = ROOT_DIR / 'cache' / 'docs'
CODE_EMBEDDINGS_CACHE_DIR = ROOT_DIR / 'cache' / 'code_embeddings'
ANNS_CACHE_DIR = ROOT_DIR / 'cache' / 'anns'

SEQS_CACHE_FILENAME = '{language}_{set_}_{type_}.npy'
MODEL_CACHE_FILENAME = '{language}.hdf5'
VOCABULARY_CACHE_FILENAME = '{language}_{type_}.pkl'
DOCS_CACHE_FILENAME = '{language}_{set_}.jsonl'
CODE_EMBEDDINGS_CACHE_FILENAME = '{language}.npy'
ANN_CACHE_FILENAME = '{language}.ann'

LANGUAGES_NUM_FILES = {
    'python': 14,
    # 'ruby': 2,
    # 'php': 18,
    # 'go': 11,
    # 'javascript': 5,
    # 'java': 16,
}
LANGUAGES = list(sorted(LANGUAGES_NUM_FILES.keys()))
DATA_SETS = ['train', 'valid', 'test']

# Model constants
TRAIN_BATCH_SIZE = 1000
EMBEDDING_SIZE = 256
LEARNING_RATE = 0.01

FUNC_NAME_AS_QUERY_PCT = 0.1
MIN_FUNC_NAME_QUERY_LENGTH = 12
VOCABULARY_PCT_BPE = 0.5

CODE_MAX_SEQ_LENGTH = 30 if SIAMESE_MODEL else 200
CODE_VOCABULARY_SIZE = 10000
CODE_TOKEN_COUNT_THRESHOLD = 10

QUERY_MAX_SEQ_LENGTH = 30
QUERY_VOCABULARY_SIZE = 10000
QUERY_TOKEN_COUNT_THRESHOLD = 10

SIAMESE_MAX_SEQ_LENGTH = 30
SIAMESE_VOCABULARY_SIZE = 10000

ENV = {
    "RESOURCES_DIR": "/datadrive/codesnippetsearch/resources/",
    "DATA_DIR": "/datadrive/codesnippetsearch/resources/data/",
}

RATE_DROP_DENSE = 0.25
NUMBER_DENSE_UNITS = 256

CONFIG = {
    'learning_rate': LEARNING_RATE,
    'train_batch_size': TRAIN_BATCH_SIZE,
    'embedding_size': EMBEDDING_SIZE,
    'func_name_as_query_pct': FUNC_NAME_AS_QUERY_PCT,
    'min_func_name_query_length': MIN_FUNC_NAME_QUERY_LENGTH,
    'vocabulary_pct_bpe': VOCABULARY_PCT_BPE,
    'code_vocabulary_size': CODE_VOCABULARY_SIZE,
    'code_token_count_threshold': CODE_TOKEN_COUNT_THRESHOLD,
    'code_max_seq_length': CODE_MAX_SEQ_LENGTH,
    'query_vocabulary_size': QUERY_VOCABULARY_SIZE,
    'query_token_count_threshold': QUERY_TOKEN_COUNT_THRESHOLD,
    'query_max_seq_length': QUERY_MAX_SEQ_LENGTH,

    'rate_drop_dense': RATE_DROP_DENSE,
    'number_dense_units': NUMBER_DENSE_UNITS
}
