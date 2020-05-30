import random

import numpy as np
import tensorflow as tf
from tensorflow.keras import backend
from tensorflow.keras.layers import Dropout, Embedding, Input, Lambda, Layer
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Nadam
from wandb.keras import WandbCallback

from rok import evaluate_model, utils, shared
from rok.attention import SelfAttention


# mm model
def get_mm_model():
    inputs = list()
    embeddings = list()
    for data_type in shared.SUB_TYPES:
        input_data, embedding_data = get_embedding_layer(data_type)
        inputs.append(input_data)
        embeddings.append(embedding_data)
    code_embedding, query_embedding = utils.repack_embeddings(embeddings)
    merge_layer = Lambda(cosine_similarity, name='cosine_similarity')(
        [code_embedding, query_embedding]
    )

    model = Model(inputs=inputs, outputs=merge_layer)
    model.compile(optimizer=Nadam(), loss=cosine_loss)
    return model


# siamese model
def get_siamese_base_model(input_shape) -> Model:
    siamese_input = Input(shape=input_shape, name='siamese_input')
    siamese_embedding = Embedding(
        input_length=shared.SIAMESE_MAX_SEQ_LEN,
        input_dim=shared.VOCAB_SIZE,
        output_dim=shared.EMBEDDING_SIZE,
        name='siamese_embedding',
        mask_zero=True)(siamese_input)
    siamese_embedding = ZeroMaskedEntries()(siamese_embedding)
    # siamese_embedding = Dropout(0.5)(siamese_embedding)
    siamese_embedding = Lambda(
        mask_aware_mean, mask_aware_mean_output_shape, name='siamese_embedding_mean'
    )(siamese_embedding)
    siamese_base_model = Model(
        inputs=siamese_input, outputs=siamese_embedding, name='base_model'
    )
    return siamese_base_model


def get_siamese_head_model(embedding_shape) -> Model:
    code_embedding = Input(shape=embedding_shape)
    query_embedding = Input(shape=embedding_shape)
    siamese_score = Lambda(cosine_similarity, name='cosine_similarity')(
        [code_embedding, query_embedding]
    )

    siamese_head_model = Model(
        inputs=[code_embedding, query_embedding], outputs=siamese_score, name='head_model')
    return siamese_head_model


def get_siamese_model() -> Model:
    # https://github.com/aspamers/siamese
    # https://keras.io/examples/mnist_siamese/
    input_shape = (shared.SIAMESE_MAX_SEQ_LEN,)
    siamese_base_model = get_siamese_base_model(input_shape)
    siamese_head_model = get_siamese_head_model(siamese_base_model.output_shape)

    code_input = Input(shape=input_shape, name='code_input')
    query_input = Input(shape=input_shape, name='query_input')
    code_embedding = siamese_base_model(code_input)
    query_embedding = siamese_base_model(query_input)
    siamese_score = siamese_head_model([code_embedding, query_embedding])

    siamese_model = Model(inputs=[code_input, query_input], outputs=siamese_score)
    siamese_model.compile(optimizer=Nadam(), loss=cosine_loss)
    return siamese_model


# vanilla model
def get_vanilla_model() -> Model:
    code_input, code_embedding = get_embedding_layer('code')
    query_input, query_embedding = get_embedding_layer('query')
    merge_layer = Lambda(cosine_similarity, name='cosine_similarity')(
        [code_embedding, query_embedding]
    )

    model = Model(inputs=[code_input, query_input], outputs=merge_layer)
    model.compile(optimizer=Nadam(), loss=cosine_loss)
    return model


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


def get_embedding_layer(data_type: str):
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
        # embeddings = Dropout(0.5)(embeddings)
    embeddings = Lambda(
        mask_aware_mean, mask_aware_mean_output_shape, name=f'{data_type}_embedding_mean')(embeddings)

    return inputs, embeddings


def get_embedding_predictor(model, data_type: str):
    if shared.SIAMESE:
        base_model = model.get_layer('base_model')
        inputs = base_model.get_layer('siamese_input').input
        outputs = base_model.get_layer('siamese_embedding_mean').output
    else:
        inputs = model.get_layer(f'{data_type}_input').input
        outputs = model.get_layer(f'{data_type}_embedding_mean').output
    return Model(inputs=inputs, outputs=outputs)


def get_model() -> Model:
    if shared.MM:
        model = get_mm_model()
    elif shared.SIAMESE:
        model = get_siamese_model()
    else:
        model = get_vanilla_model()
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
            batch_seqs_dict.setdefault(f'{data_type}_input', batch_encoded_seqs)
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
    # recreate the masks - all zero rows have been masked
    mask = backend.not_equal(backend.sum(backend.abs(inputs), axis=2, keepdims=True), 0)
    # number of that rows are not all zeros
    n = backend.sum(backend.cast(mask, 'float32'), axis=1, keepdims=False)
    # compute mask-aware mean of x
    inputs_mean = backend.sum(inputs, axis=1, keepdims=False) / (n + 1E-8)

    return inputs_mean


def mask_aware_mean_output_shape(input_shape):
    shape = list(input_shape)
    assert len(shape) == 3
    return shape[0], shape[2]


def training(language, verbose=True):
    model = get_model()

    train_seqs_dict = dict()
    valid_seqs_list = list()
    for data_type in shared.SUB_TYPES:
        train_seqs = utils.load_seqs(language, 'train', data_type)
        train_seqs_dict.setdefault(data_type, train_seqs)
        valid_seqs = utils.load_seqs(language, 'valid', data_type)
        valid_seqs_list.append(valid_seqs)

    n_samples = list(train_seqs_dict.values())[0].shape[0]
    batch_generator = generate_batch(train_seqs_dict, batch_size=shared.BATCH_SIZE)

    additional_callback = [WandbCallback(monitor='val_loss', save_model=False)] if shared.WANDB else []
    model.fit(
        batch_generator, epochs=200,
        steps_per_epoch=n_samples // shared.BATCH_SIZE,
        verbose=2 if verbose else -1,
        callbacks=[evaluate_model.MrrEarlyStopping(valid_seqs_list)] + additional_callback
    )

    utils.save_model(language, model)
