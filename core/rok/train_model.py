import argparse
import random

import numpy as np
import tensorflow as tf
import wandb
from keras.engine import Layer
from keras import backend
from keras.callbacks import EarlyStopping
from keras.layers import Embedding, Input, Lambda
from keras.models import Model
from keras.optimizers import Adam, Nadam
from wandb.keras import WandbCallback

from rok import evaluate_model, utils
from rok import shared

np.random.seed(0)
random.seed(0)


class MrrEarlyStopping(EarlyStopping):
    def __init__(self,
                 padded_encoded_code_validation_seqs,
                 padded_encoded_query_validation_seqs,
                 patience=5,
                 batch_size=1000):
        super().__init__(monitor='val_mrr', mode='max', restore_best_weights=True, verbose=True, patience=patience)
        self.padded_encoded_code_validation_seqs = padded_encoded_code_validation_seqs
        self.padded_encoded_query_validation_seqs = padded_encoded_query_validation_seqs
        self.batch_size = batch_size

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}

        mean_mrr = evaluate_model.evaluate_model_mean_mrr(
            self.model,
            self.padded_encoded_code_validation_seqs,
            self.padded_encoded_query_validation_seqs,
            batch_size=self.batch_size)

        print('Mean MRR:', mean_mrr)
        super().on_epoch_end(epoch, {**logs, 'val_mrr': mean_mrr})


def get_code_input_and_embedding_layer():
    code_input = Input(shape=(shared.CODE_MAX_SEQ_LENGTH,), name='code_input')
    code_embedding = Embedding(
        input_length=shared.CODE_MAX_SEQ_LENGTH,
        input_dim=shared.CODE_VOCABULARY_SIZE,
        output_dim=shared.EMBEDDING_SIZE,
        name='code_embedding',
        mask_zero=True)(code_input)
    code_embedding = ZeroMaskedEntries()(code_embedding)
    code_embedding = Lambda(
        mask_aware_mean, mask_aware_mean_output_shape, name='code_embedding_mean')(code_embedding)

    return code_input, code_embedding


def get_query_input_and_embedding_layer():
    query_input = Input(shape=(shared.QUERY_MAX_SEQ_LENGTH,), name='query_input')
    query_embedding = Embedding(
        input_length=shared.QUERY_MAX_SEQ_LENGTH,
        input_dim=shared.QUERY_VOCABULARY_SIZE,
        output_dim=shared.EMBEDDING_SIZE,
        name='query_embedding',
        mask_zero=True)(query_input)
    query_embedding = ZeroMaskedEntries()(query_embedding)
    query_embedding = Lambda(
        mask_aware_mean, mask_aware_mean_output_shape, name='query_embedding_mean')(query_embedding)

    return query_input, query_embedding


def get_code_embedding_predictor(model):
    if shared.SIAMESE_MODEL:
        base_model = model.get_layer('base_model')
        code_input = base_model.get_layer('siamese_input').input
        code_output = base_model.get_layer('siamese_embedding_mean').output
    else:
        code_input = model.get_layer('code_input').input
        code_output = model.get_layer('code_embedding_mean').output
    return Model(inputs=code_input, outputs=code_output)


def get_query_embedding_predictor(model):
    if shared.SIAMESE_MODEL:
        base_model = model.get_layer('base_model')
        query_input = base_model.get_layer('siamese_input').input
        query_output = base_model.get_layer('siamese_embedding_mean').output
    else:
        query_input = model.get_layer('query_input').input
        query_output = model.get_layer('query_embedding_mean').output
    return Model(inputs=query_input, outputs=query_output)


def get_siamese_base_model(input_shape) -> Model:
    siamese_input = Input(shape=input_shape, name='siamese_input')
    siamese_embedding = Embedding(
        input_length=shared.SIAMESE_MAX_SEQ_LENGTH,
        input_dim=shared.SIAMESE_VOCABULARY_SIZE,
        output_dim=shared.EMBEDDING_SIZE,
        name='siamese_embedding',
        mask_zero=True)(siamese_input)
    siamese_embedding = ZeroMaskedEntries()(siamese_embedding)
    siamese_embedding = Lambda(
        mask_aware_mean, mask_aware_mean_output_shape, name='siamese_embedding_mean'
    )(siamese_embedding)
    siamese_base_model = Model(inputs=siamese_input, outputs=siamese_embedding, name='base_model')
    return siamese_base_model


def get_siamese_head_model(embedding_shape) -> Model:
    code_embedding = Input(shape=embedding_shape)
    query_embedding = Input(shape=embedding_shape)
    siamese_score = Lambda(cosine_similarity, name='cosine_similarity')(
        [code_embedding, query_embedding]
    )

    siamese_head_model = Model(inputs=[code_embedding, query_embedding], outputs=siamese_score, name='head_model')
    return siamese_head_model


def get_siamese_model() -> Model:
    # https://github.com/aspamers/siamese
    # https://keras.io/examples/mnist_siamese/
    input_shape = (shared.SIAMESE_MAX_SEQ_LENGTH,)
    siamese_base_model = get_siamese_base_model(input_shape)
    siamese_head_model = get_siamese_head_model(siamese_base_model.output_shape)

    code_input = Input(shape=(shared.SIAMESE_MAX_SEQ_LENGTH,), name='code_input')
    query_input = Input(shape=(shared.SIAMESE_MAX_SEQ_LENGTH,), name='query_input')
    code_embedding = siamese_base_model(code_input)
    query_embedding = siamese_base_model(query_input)
    siamese_score = siamese_head_model([code_embedding, query_embedding])

    siamese_model = Model(inputs=[code_input, query_input], outputs=siamese_score)
    # siamese_model.compile(optimizer=Adam(learning_rate=shared.LEARNING_RATE), loss=cosine_loss)
    siamese_model.compile(optimizer=Nadam(), loss=cosine_loss)
    return siamese_model


def create_siamese_pairs(batch_code_seqs, batch_query_seqs, num_samples):
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


def generate_siamese_batch(padded_encoded_code_seqs, padded_encoded_query_seqs, batch_size: int):
    n_samples = padded_encoded_code_seqs.shape[0]

    shuffled_indices = np.arange(0, n_samples)
    np.random.shuffle(shuffled_indices)
    padded_encoded_code_seqs = padded_encoded_code_seqs[shuffled_indices, :]
    padded_encoded_query_seqs = padded_encoded_query_seqs[shuffled_indices, :]

    idx = 0
    while True:
        end_idx = min(idx + batch_size, n_samples)
        n_batch_samples = min(batch_size, end_idx - idx)

        batch_code_seqs = padded_encoded_code_seqs[idx:end_idx, :]
        batch_query_seqs = padded_encoded_query_seqs[idx:end_idx, :]
        pairs, labels = create_siamese_pairs(batch_code_seqs, batch_query_seqs, n_batch_samples)
        # The siamese network expects two inputs and one output. Split the pairs into a list of inputs.
        # yield [pairs[:, 0], pairs[:, 1]], labels
        yield {'code_input': batch_code_seqs, 'query_input': batch_query_seqs}, np.zeros(n_batch_samples)

        idx += n_batch_samples
        if idx >= n_samples:
            idx = 0


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


def get_vanilla_model() -> Model:
    code_input, code_embedding = get_code_input_and_embedding_layer()
    query_input, query_embedding = get_query_input_and_embedding_layer()
    merge_layer = Lambda(cosine_similarity, name='cosine_similarity')(
        [code_embedding, query_embedding]
    )

    model = Model(inputs=[code_input, query_input], outputs=merge_layer)
    model.compile(optimizer=Adam(learning_rate=shared.LEARNING_RATE), loss=cosine_loss)
    return model


def get_model() -> Model:
    if shared.SIAMESE_MODEL:
        model = get_siamese_model()
    else:
        model = get_vanilla_model()
    return model


def generate_batch(padded_encoded_code_seqs, padded_encoded_query_seqs, batch_size: int):
    n_samples = padded_encoded_code_seqs.shape[0]

    shuffled_indices = np.arange(0, n_samples)
    np.random.shuffle(shuffled_indices)
    padded_encoded_code_seqs = padded_encoded_code_seqs[shuffled_indices, :]
    padded_encoded_query_seqs = padded_encoded_query_seqs[shuffled_indices, :]

    idx = 0
    while True:
        end_idx = min(idx + batch_size, n_samples)
        n_batch_samples = min(batch_size, end_idx - idx)

        batch_code_seqs = padded_encoded_code_seqs[idx:end_idx, :]
        batch_query_seqs = padded_encoded_query_seqs[idx:end_idx, :]
        yield {'code_input': batch_code_seqs, 'query_input': batch_query_seqs}, np.zeros(n_batch_samples)

        idx += n_batch_samples
        if idx >= n_samples:
            idx = 0


def train(language, model_callbacks, verbose=True):
    model = get_model()

    train_code_seqs = utils.load_cached_seqs(language, 'train', 'code')
    train_query_seqs = utils.load_cached_seqs(language, 'train', 'query')

    valid_code_seqs = utils.load_cached_seqs(language, 'valid', 'code')
    valid_query_seqs = utils.load_cached_seqs(language, 'valid', 'query')

    num_samples = train_code_seqs.shape[0]
    if shared.SIAMESE_MODEL:
        model.fit_generator(
            generate_siamese_batch(train_code_seqs, train_query_seqs, batch_size=shared.TRAIN_BATCH_SIZE),
            epochs=200,
            steps_per_epoch=num_samples // shared.TRAIN_BATCH_SIZE,
            verbose=2 if verbose else -1,
            callbacks=[
                MrrEarlyStopping(valid_code_seqs, valid_query_seqs, patience=5)
            ] + model_callbacks
        )
    else:
        model.fit_generator(
            generate_batch(train_code_seqs, train_query_seqs, batch_size=shared.TRAIN_BATCH_SIZE),
            epochs=200,
            steps_per_epoch=num_samples // shared.TRAIN_BATCH_SIZE,
            verbose=2 if verbose else -1,
            callbacks=[
                MrrEarlyStopping(valid_code_seqs, valid_query_seqs, patience=5)
            ] + model_callbacks
        )

    model.save(utils.get_cached_model_path(language))


def main():
    parser = argparse.ArgumentParser(description='Train individual language models from prepared data.')
    parser.add_argument('--notes', default='')
    utils.add_bool_arg(parser, 'wandb', default=True)
    utils.add_bool_arg(parser, 'evaluate', default=True)
    args = vars(parser.parse_args())

    if args['wandb']:
        wandb.init(project='CodeSearchNet', notes=args['notes'], config=shared.CONFIG)
        additional_callbacks = [WandbCallback(monitor='val_loss', save_model=False)]
    else:
        additional_callbacks = []

    for language in shared.LANGUAGES:
        print(f'Training {language}')
        train(language, additional_callbacks)

    if args['evaluate']:
        evaluate_model.evaluate_mean_mrr(args['wandb'])
        evaluate_model.emit_ndcg_model_predictions(args['wandb'])


class ZeroMaskedEntries(Layer):
    def __init__(self, **kwargs):
        self.support_mask = True
        super(ZeroMaskedEntries, self).__init__(**kwargs)

    def build(self, input_shape):
        self.output_dim = input_shape[1]
        self.repeat_dim = input_shape[2]

    def call(self, x, mask=None):
        mask = backend.cast(mask, 'float32')
        mask = backend.repeat(mask, self.repeat_dim)
        mask = backend.permute_dimensions(mask, (0, 2, 1))
        return x * mask

    def compute_mask(self, input_shape, input_mask=None):
        return None


def mask_aware_mean(x):
    # recreate the masks - all zero rows have been masked
    mask = backend.not_equal(backend.sum(backend.abs(x), axis=2, keepdims=True), 0)

    # number of that rows are not all zeros
    n = backend.sum(backend.cast(mask, 'float32'), axis=1, keepdims=False)

    # compute mask-aware mean of x
    x_mean = backend.sum(x, axis=1, keepdims=False) / n

    return x_mean


def mask_aware_mean_output_shape(input_shape):
    shape = list(input_shape)
    assert len(shape) == 3
    return shape[0], shape[2]


if __name__ == '__main__':
    main()
