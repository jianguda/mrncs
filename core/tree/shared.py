import random
from enum import Enum
from pathlib import Path

import numpy as np


class ModeEnum(Enum):
    # NBoW
    CODE = 0  # 200
    ROOTPATH = 1  # 200
    LEAFPATH = 2  # 200
    SBT = 3  # 200
    LCRS = 4  # 200
    # MM
    MM_CODE_ROOTPATH = 11  # 200
    MM_CODE_LEAFPATH = 12  # 200
    MM_CODE_SBT = 13  # 200
    MM_CODE_LCRS = 14  # 200
    MM_CODE_ROOTPATH_SBT = 25  # 200


BERT1 = False  # use bert1 (complex) or bert2 (simple)
CONTEXT = True  # BERT if CONTEXT else NBoW
LAZY = False  # use existing models and evaluation embeddings when available
WANDB = True
ANNOY = False  # recommend: False (KNN)
ATTENTION = False  # tip (run evaluation separately after training)
PROCESSING = True  # recommend: True
DESENSITIZE = False  # recommend: False
# DATA_ENHANCEMENT = False

MODE = ModeEnum.CODE if PROCESSING else ModeEnum.CODE
MODE_TAG = MODE.name.lower()

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
SEQ_FILENAME = '{language}_{data_set}_{data_type}.npy'
MODEL_FILENAME = '{language}_{mode_tag}.hdf5'
EMBEDDING_FILENAME = '{language}_{data_type}_{mode_tag}.npy'

encoded_pads_dict = dict()
# only 1 language should be chosen each time
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
DATA_TYPES = ['code', 'rootpath', 'leafpath', 'sbt', 'lcrs', 'query'] if PROCESSING else ['code', 'query']
if MODE is ModeEnum.ROOTPATH:
    SUB_TYPES = ['rootpath', 'query']
elif MODE is ModeEnum.LEAFPATH:
    SUB_TYPES = ['leafpath', 'query']
elif MODE is ModeEnum.SBT:
    SUB_TYPES = ['sbt', 'query']
elif MODE is ModeEnum.LCRS:
    SUB_TYPES = ['lcrs', 'query']
elif MODE is ModeEnum.MM_CODE_ROOTPATH:
    SUB_TYPES = ['code', 'rootpath', 'query']
elif MODE is ModeEnum.MM_CODE_LEAFPATH:
    SUB_TYPES = ['code', 'leafpath', 'query']
elif MODE is ModeEnum.MM_CODE_SBT:
    SUB_TYPES = ['code', 'sbt', 'query']
elif MODE is ModeEnum.MM_CODE_LCRS:
    SUB_TYPES = ['code', 'lcrs', 'query']
else:  # ModeEnum.CODE
    SUB_TYPES = ['code', 'query']

BATCH_SIZE = 450 if CONTEXT else 1000
LEARNING_RATE = 5e-4 if CONTEXT else 0.01
EMBEDDING_SIZE = 128

VOCAB_PCT_BPE = 0.5
VOCAB_SIZE = 10000

CODE_SEQ_LEN = 200
QUERY_SEQ_LEN = 30

random.seed(0)
np.random.seed(0)
