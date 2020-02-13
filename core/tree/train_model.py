import random

import numpy as np
import tensorflow as tf
from tensorflow.keras import backend
from tensorflow.keras.layers import Dropout, Embedding, Input, Lambda, Layer
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Nadam
from wandb.keras import WandbCallback

from tree import evaluate_model, utils, shared
from tree.attention import SelfAttention
# pip install bert-for-tf2
from bert import BertModelLayer


def create_enhanced_pairs(batch_code_seqs, batch_query_seqs, num_samples):
    positive_pairs = []
    positive_labels = []
    # create_positive_pairs
    for sample_idx in range(num_samples):
        positive_pairs.append([batch_code_seqs[sample_idx], batch_query_seqs[sample_idx]])
        positive_labels.append([0.0])

    negative_pairs = []
    negative_labels = []
    # create_negative_pairs
    for sample_idx in range(num_samples):
        unique_idx = sample_idx
        while unique_idx == sample_idx:
            unique_idx = random.randint(0, num_samples - 1)
        negative_pairs.append([batch_code_seqs[sample_idx], batch_query_seqs[unique_idx]])
        negative_labels.append([1.0])

    return np.array(positive_pairs + negative_pairs), np.array(positive_labels + negative_labels)


def cosine_similarity(x):
    code_embedding, query_embedding = x
    query_norms = tf.norm(query_embedding, axis=-1, keepdims=True) + 1e-10
    code_norms = tf.norm(code_embedding, axis=-1, keepdims=True) + 1e-10
    return tf.matmul(
        query_embedding / query_norms, code_embedding / code_norms, transpose_a=False, transpose_b=True)


def cosine_loss(_, cosine_similarity_matrix):
    neg_matrix = tf.linalg.diag(tf.fill(dims=[tf.shape(cosine_similarity_matrix)[0]], value=float('-inf')))

    # Distance between query and code snippet should be as small as possible
    diagonal_cosine_distance = 1. - tf.linalg.diag_part(cosine_similarity_matrix)
    # Max. similarity between query and non-corresponding code snippet should be as small as possible
    max_positive_non_diagonal_similarity_in_row = tf.reduce_max(
        tf.nn.relu(cosine_similarity_matrix + neg_matrix), axis=-1)

    # Combined distance and similarity should be as small as possible as well
    per_sample_loss = tf.maximum(0., diagonal_cosine_distance + max_positive_non_diagonal_similarity_in_row)
    return tf.reduce_mean(per_sample_loss)


def get_bert_embedding_layer(data_type: str):
    l_bert = BertModelLayer(**BertModelLayer.Params(
        vocab_size=10000,  # embedding params
        use_token_type=True,
        use_position_embeddings=True,
        token_type_vocab_size=2,

        num_layers=12,  # transformer encoder params
        hidden_size=768,
        hidden_dropout=0.1,
        intermediate_size=4 * 768,
        intermediate_activation="gelu",

        adapter_size=None,  # see arXiv:1902.00751 (adapter-BERT)
        shared_layer=False,  # True for ALBERT (arXiv:1909.11942)
        embedding_size=None,  # None for BERT, wordpiece embedding size for ALBERT

        name="bert"  # any other Keras layer params
    ))

    input_length = utils.get_input_length(data_type)
    inputs = Input(shape=(input_length,), name=f'{data_type}_input', dtype='int32')

    # using the default token_type/segment id 0
    # output: [batch_size, max_seq_len, hidden_size]
    embeddings = l_bert(inputs, name=f'{data_type}_embedding_mean')

    return inputs, embeddings


def get_nbow_embedding_layer(data_type: str):
    input_length = utils.get_input_length(data_type)
    inputs = Input(shape=(input_length,), name=f'{data_type}_input')
    embeddings = Embedding(
        input_length=input_length,
        input_dim=shared.VOCAB_SIZE,
        output_dim=shared.EMBEDDING_SIZE,
        name=f'{data_type}_embedding',
        mask_zero=True)(inputs)
    embeddings = ZeroMaskedEntries()(embeddings)
    # embeddings = Dropout(0.5)(embeddings)
    if shared.ATTENTION:
        embeddings = SelfAttention(16, shared.EMBEDDING_SIZE)(embeddings)
    embeddings = Lambda(
        mask_aware_mean, mask_aware_mean_shape, name=f'{data_type}_embedding_mean')(embeddings)
    return inputs, embeddings


def get_embedding_predictor(model, data_type: str):
    inputs = model.get_layer(f'{data_type}_input').input
    outputs = model.get_layer(f'{data_type}_embedding_mean').output
    return Model(inputs=inputs, outputs=outputs)


def get_model() -> Model:
    inputs = list()
    embeddings = list()
    for data_type in shared.SUB_TYPES:
        if shared.CONTEXT:
            input_data, embedding_data = get_bert_embedding_layer(data_type)
        else:
            input_data, embedding_data = get_nbow_embedding_layer(data_type)
        inputs.append(input_data)
        embeddings.append(embedding_data)
    embeddings = utils.repack_embeddings(embeddings)

    merge_layer = Lambda(cosine_similarity, name='cosine_similarity')(embeddings)
    model = Model(inputs=inputs, outputs=merge_layer)
    model.compile(optimizer=Nadam(), loss=cosine_loss)
    return model


def generate_batch(train_seqs_dict, batch_size: int):
    n_samples = list(train_seqs_dict.values())[0].shape[0]

    shuffled_indices = np.arange(0, n_samples)
    np.random.shuffle(shuffled_indices)
    for data_value in train_seqs_dict.keys():
        train_seqs = train_seqs_dict.get(data_value)
        train_seqs = train_seqs[shuffled_indices, :]
        train_seqs_dict[data_value] = train_seqs

    idx = 0
    while True:
        end_idx = min(idx + batch_size, n_samples)
        n_batch_samples = min(batch_size, end_idx - idx)

        batch_seqs_dict = dict()
        for data_type, encoded_seqs in train_seqs_dict.items():
            batch_encoded_seqs = encoded_seqs[idx:end_idx, :]
            batch_seqs_dict[f'{data_type}_input'] = batch_encoded_seqs
        # if shared.DATA_ENHANCEMENT:
        #     pairs, labels = create_enhanced_pairs(batch_code_seqs, batch_query_seqs, n_batch_samples)
        #     yield [pairs[:, 0], pairs[:, 1]], labels
        # else:
        yield batch_seqs_dict, np.zeros(n_batch_samples)

        idx += n_batch_samples
        if idx >= n_samples:
            idx = 0


class ZeroMaskedEntries(Layer):
    def __init__(self, **kwargs):
        self.support_mask = True
        super(ZeroMaskedEntries, self).__init__(**kwargs)

    def build(self, input_shape):
        self.output_dim = input_shape[1]
        self.repeat_dim = input_shape[2]

    def call(self, inputs, mask=None):
        mask = backend.cast(mask, 'float32')
        mask = backend.repeat(mask, self.repeat_dim)
        mask = backend.permute_dimensions(mask, (0, 2, 1))
        return inputs * mask

    def compute_mask(self, input_shape, input_mask=None):
        return None


def mask_aware_mean(inputs):
    # https://github.com/github/CodeSearchNet/blob/master/src/utils/tfutils.py#L107
    # recreate the masks - all zero rows have been masked
    mask = backend.not_equal(backend.sum(backend.abs(inputs), axis=2, keepdims=True), 0)
    # number of that rows are not all zeros
    num = backend.sum(backend.cast(mask, 'float32'), axis=1, keepdims=False)
    # compute mask-aware mean of inputs
    inputs_mean = backend.sum(inputs, axis=1, keepdims=False) / (num + 1E-8)

    return inputs_mean


def mask_aware_mean_shape(input_shape):
    shape = list(input_shape)
    assert len(shape) == 3
    return shape[0], shape[2]


def train_model(language: str, train_seqs_dict: dict, valid_seqs_dict: dict):
    if utils.check_model(language):
        return
    model = get_model()
    n_samples = list(train_seqs_dict.values())[0].shape[0]
    batch_generator = generate_batch(train_seqs_dict, batch_size=shared.BATCH_SIZE)
    additional_callback = [WandbCallback(monitor='val_loss', save_model=False)] if shared.WANDB else []
    model.fit(
        batch_generator, epochs=200, verbose=2,
        steps_per_epoch=n_samples // shared.BATCH_SIZE,
        callbacks=[evaluate_model.MrrEarlyStopping(valid_seqs_dict)] + additional_callback
    )
    utils.save_model(language, model)


def training():
    print('Training')
    general_train_seqs_dict = dict()
    general_valid_seqs_dict = dict()
    for language in shared.LANGUAGES:
        train_seqs_dict = dict()
        valid_seqs_dict = dict()
        for data_type in shared.SUB_TYPES:
            # train_seqs
            train_seqs = utils.load_seq(language, 'train', data_type)
            train_seqs_dict[data_type] = train_seqs
            if data_type in general_train_seqs_dict:
                general_train_seqs = general_train_seqs_dict[data_type]
                train_seqs = np.vstack((general_train_seqs, train_seqs))
            general_train_seqs_dict[data_type] = train_seqs
            # valid_seqs
            valid_seqs = utils.load_seq(language, 'valid', data_type)
            valid_seqs_dict[data_type] = valid_seqs
            if data_type in general_valid_seqs_dict:
                general_valid_seqs = general_valid_seqs_dict[data_type]
                valid_seqs = np.vstack((general_valid_seqs, valid_seqs))
            general_valid_seqs_dict[data_type] = valid_seqs
        # Check for invalid sequences when it is not for evaluation
        train_seqs_dict = utils.filter_valid_seqs(train_seqs_dict)
        valid_seqs_dict = utils.filter_valid_seqs(valid_seqs_dict)
        if not shared.GENERAL:
            train_model(language, train_seqs_dict, valid_seqs_dict)
    if shared.GENERAL:
        train_model('general', general_train_seqs_dict, general_valid_seqs_dict)
