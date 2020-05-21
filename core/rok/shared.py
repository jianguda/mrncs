import random
from enum import Enum
from pathlib import Path

import numpy as np


class ModeEnum(Enum):
    NBoW = 0
    AUX = 1
    MM_PATH = 10
    MM_SBT = 11  # https://arxiv.org/abs/2005.06980
    SIAMESE_30 = 20
    SIAMESE_200 = 21


MODE = ModeEnum.MM_SBT
MODE_TAG = MODE.name.lower()
MM = MODE in (ModeEnum.MM_PATH, ModeEnum.MM_SBT)
SIAMESE = MODE in (ModeEnum.SIAMESE_30, ModeEnum.SIAMESE_200)

WANDB = True
ANNOY = True  # KNN
TURBO = True  # for quick experiments
ATTENTION = False
DATA_ENHANCEMENT = False

ROOT_DIR = Path.cwd().parent
RESOURCES_DIR = ROOT_DIR / 'resources'
DATA_DIR = RESOURCES_DIR / 'data'
CACHES_DIR = RESOURCES_DIR / 'caches'
DOCS_DIR = CACHES_DIR / 'docs'
VOCABS_DIR = CACHES_DIR / 'vocabs' / MODE_TAG
SEQS_DIR = CACHES_DIR / 'seqs' / MODE_TAG
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
DATA_TYPES = ['code', 'leaf', 'path', 'sbt', 'aux', 'query']
if MODE is ModeEnum.SIAMESE_30:
    SUB_TYPES = ['leaf', 'query']
elif MODE is ModeEnum.MM_PATH:
    SUB_TYPES = ['leaf', 'path', 'query']
elif MODE is ModeEnum.MM_SBT:
    SUB_TYPES = ['code', 'sbt', 'query']
elif MODE is ModeEnum.AUX:
    SUB_TYPES = ['aux', 'query']
else:  # ModeEnum.NBoW, ModeEnum.SIAMESE_200
    SUB_TYPES = ['code', 'query']

BATCH_SIZE = 256
EMBEDDING_SIZE = 128

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
