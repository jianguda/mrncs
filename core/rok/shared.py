import random
from enum import Enum
from pathlib import Path

import numpy as np


class ModeEnum(Enum):
    NBoW = 0
    SBT = 1
    LEAF = 2
    MM_SBT = 10  # https://arxiv.org/abs/2005.06980
    MM_PATH = 11
    SIAMESE_30 = 20
    SIAMESE_200 = 21


MODE = ModeEnum.LEAF
MODE_TAG = MODE.name.lower()
MM = MODE in (ModeEnum.MM_SBT, ModeEnum.MM_PATH)
SIAMESE = MODE in (ModeEnum.SIAMESE_30, ModeEnum.SIAMESE_200)

WANDB = True
ANNOY = False  # KNN
GENERAL = False  # for multi-lang
ATTENTION = False
DATA_ENHANCEMENT = False

ROOT_DIR = Path.cwd().parent
RESOURCES_DIR = ROOT_DIR / 'resources'
DATA_DIR = RESOURCES_DIR / 'data'
CACHES_DIR = RESOURCES_DIR / 'caches'
DOCS_DIR = CACHES_DIR / 'docs'
VOCABS_DIR = CACHES_DIR / 'vocabs'
SEQS_DIR = CACHES_DIR / 'seqs'
MODELS_DIR = CACHES_DIR / 'models' / MODE_TAG
EMBEDDINGS_DIR = CACHES_DIR / 'embeddings' / MODE_TAG

DOCS_FILENAME = '{language}_{data_set}.jsonl'
VOCABS_FILENAME = '{language}_{data_type}.pkl'
SEQS_FILENAME = '{language}_{data_set}_{data_type}.npy'
MODELS_FILENAME = '{language}.hdf5'
EMBEDDINGS_FILENAME = '{language}_{data_type}.npy'

CORPUS_FILES = {
    'python': 14,
    # 'ruby': 2,
    # 'php': 18,
    # 'go': 11,
    # 'javascript': 5,
    # 'java': 16,
}

LANGUAGES = list(sorted(CORPUS_FILES.keys()))
DATA_SETS = ['train', 'valid', 'test']
DATA_TYPES = ['code', 'leaf', 'path', 'sbt', 'query']
if MODE is ModeEnum.SBT:
    SUB_TYPES = ['sbt', 'query']
elif MODE is ModeEnum.MM_SBT:
    SUB_TYPES = ['code', 'sbt', 'query']
elif MODE is ModeEnum.MM_PATH:
    SUB_TYPES = ['code', 'path', 'query']
elif MODE in (ModeEnum.LEAF, ModeEnum.SIAMESE_30):
    SUB_TYPES = ['leaf', 'query']
else:  # ModeEnum.NBoW, ModeEnum.SIAMESE_200
    SUB_TYPES = ['code', 'query']

BATCH_SIZE = 256
EMBEDDING_SIZE = 256

VOCAB_PCT_BPE = 0.5
VOCAB_SIZE = 10000

SIAMESE_MAX_SEQ_LEN = 200 if MODE is ModeEnum.SIAMESE_200 else 30
CODE_MAX_SEQ_LEN = SIAMESE_MAX_SEQ_LEN if SIAMESE else 200
LEAF_MAX_SEQ_LEN = SIAMESE_MAX_SEQ_LEN if SIAMESE else 30
PATH_MAX_SEQ_LEN = SIAMESE_MAX_SEQ_LEN if SIAMESE else 30
SBT_MAX_SEQ_LEN = SIAMESE_MAX_SEQ_LEN if SIAMESE else 200
QUERY_MAX_SEQ_LEN = SIAMESE_MAX_SEQ_LEN if SIAMESE else 30

random.seed(0)
np.random.seed(0)
