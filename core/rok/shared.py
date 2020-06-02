import random
from enum import Enum
from pathlib import Path

import numpy as np


class ModeEnum(Enum):
    # NBoW
    CODE = 0  # 200
    LEAF = 1  # 30
    PATH = 2  # 200
    SBT = 3  # 200
    LCRS = 4  # 200
    # MM
    MM_CODE_PATH = 10  # 200
    MM_CODE_SBT = 11  # 200
    MM_CODE_LCRS = 12  # 200
    # SIAMESE
    SIAMESE_CODE = 20  # 200
    SIAMESE_LEAF = 21  # 30


LAZY = False  # use existing models and evaluation embeddings when available
WANDB = True
ANNOY = False  # recommend: False (KNN)
GENERAL = False  # for multi-lang
ATTENTION = False  # tip (run evaluation separately after training)
PROCESSING = True  # recommend: True
DESENSITIZE = False  # recommend: False
# DATA_ENHANCEMENT = False

MODE = ModeEnum.CODE if PROCESSING else ModeEnum.CODE
MODE_TAG = MODE.name.lower()
MM = MODE in (ModeEnum.MM_CODE_PATH, ModeEnum.MM_CODE_SBT, ModeEnum.MM_CODE_LCRS)
SIAMESE = MODE in (ModeEnum.SIAMESE_CODE, ModeEnum.SIAMESE_LEAF)

POSTFIX = '' if PROCESSING else '_legacy'
ROOT_DIR = Path.cwd().parent
RESOURCES_DIR = ROOT_DIR / 'resources'
DATA_DIR = RESOURCES_DIR / 'data'
CACHES_DIR = RESOURCES_DIR / 'caches'
DOCS_DIR = CACHES_DIR / 'docs'
VOCABS_DIR = CACHES_DIR / ('vocabs' + POSTFIX)
SEQS_DIR = CACHES_DIR / ('seqs' + POSTFIX)
MODELS_DIR = CACHES_DIR / ('models' + POSTFIX)
EMBEDDINGS_DIR = CACHES_DIR / ('embeddings' + POSTFIX)

DOC_FILENAME = '{language}_{data_set}.jsonl'
VOCAB_FILENAME = '{language}_{data_type}.pkl'
SEQ_FILENAME = '{language}_{data_set}_{data_type}_{mode_tag}.npy'
MODEL_FILENAME = '{language}_{mode_tag}.hdf5'
EMBEDDING_FILENAME = '{language}_{data_type}_{mode_tag}.npy'

CORPUS_FILES = {
    # 'python': 14,
    'ruby': 2,
    # 'php': 18,
    # 'go': 11,
    # 'javascript': 5,
    # 'java': 16,
}

LANGUAGES = list(sorted(CORPUS_FILES.keys()))
DATA_SETS = ['train', 'valid', 'test']
DATA_TYPES = ['code', 'leaf', 'path', 'sbt', 'lcrs', 'query'] if PROCESSING else ['code', 'query']
if MODE is ModeEnum.PATH:
    SUB_TYPES = ['path', 'query']
elif MODE is ModeEnum.SBT:
    SUB_TYPES = ['sbt', 'query']
elif MODE is ModeEnum.LCRS:
    SUB_TYPES = ['lcrs', 'query']
elif MODE is ModeEnum.MM_CODE_PATH:
    SUB_TYPES = ['code', 'path', 'query']
elif MODE is ModeEnum.MM_CODE_SBT:
    SUB_TYPES = ['code', 'sbt', 'query']
elif MODE is ModeEnum.MM_CODE_LCRS:
    SUB_TYPES = ['code', 'lcrs', 'query']
elif MODE in (ModeEnum.LEAF, ModeEnum.SIAMESE_LEAF):
    SUB_TYPES = ['leaf', 'query']
else:  # ModeEnum.CODE, ModeEnum.SIAMESE_CODE
    SUB_TYPES = ['code', 'query']

BATCH_SIZE = 256
EMBEDDING_SIZE = 256

VOCAB_PCT_BPE = 0.5
VOCAB_SIZE = 10000

SIAMESE_SEQ_LEN = 200 if MODE is ModeEnum.SIAMESE_CODE else 30
CODE_SEQ_LEN = SIAMESE_SEQ_LEN if SIAMESE else 200
LEAF_SEQ_LEN = SIAMESE_SEQ_LEN if SIAMESE else 30
PATH_SEQ_LEN = SIAMESE_SEQ_LEN if SIAMESE else 200
SBT_SEQ_LEN = SIAMESE_SEQ_LEN if SIAMESE else 200
LCRS_SEQ_LEN = SIAMESE_SEQ_LEN if SIAMESE else 200
QUERY_SEQ_LEN = SIAMESE_SEQ_LEN if SIAMESE else 30

random.seed(0)
np.random.seed(0)
