import random

import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from tensorflow.keras import backend
from tensorflow.keras.layers import Dropout, Embedding, Input, Lambda, Layer, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from wandb.keras import WandbCallback

from tree import evaluate_model, utils, shared
from tree.attention import SelfAttention

from bert import BertModelLayer
# from bert_self_attention import BertConfig, BertModel
# from utils.tfutils import pool_sequence_embedding


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


# src/models/model.py@305
def softmax_similarity(x):
    code_embedding, query_embedding = x
    return tf.matmul(query_embedding, code_embedding, transpose_a=False, transpose_b=True)


def softmax_loss(_, similarity_matrix):
    per_sample_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=tf.range(tf.shape(similarity_matrix)[0]),  # [0, 1, 2, 3, ..., n]
        logits=similarity_matrix
    )
    return tf.reduce_mean(per_sample_loss)


# src/models/model.py@319
def cosine_similarity(x):
    code_embedding, query_embedding = x
    query_norms = tf.norm(query_embedding, axis=-1, keepdims=True) + 1e-10
    code_norms = tf.norm(code_embedding, axis=-1, keepdims=True) + 1e-10
    return tf.matmul(
        query_embedding / query_norms, code_embedding / code_norms, transpose_a=False, transpose_b=True)


def cosine_loss(_, similarity_matrix):
    neg_matrix = tf.linalg.diag(tf.fill(dims=[tf.shape(similarity_matrix)[0]], value=float('-inf')))
    # Distance between query and code snippet should be as small as possible
    diagonal_cosine_distance = 1. - tf.linalg.diag_part(similarity_matrix)
    # Max. similarity between query and non-corresponding code snippet should be as small as possible
    max_positive_non_diagonal_similarity_in_row = tf.reduce_max(tf.nn.relu(similarity_matrix + neg_matrix), axis=-1)
    # Combined distance and similarity should be as small as possible as well
    per_sample_loss = tf.maximum(0., diagonal_cosine_distance + max_positive_non_diagonal_similarity_in_row)
    return tf.reduce_mean(per_sample_loss)


# src/models/model.py@336
def maxmargin_similarity(x):
    pass


def maxmargin_loss(_, similarity_matrix):
    pass


# src/models/model.py@350
def triplet_similarity(x):
    pass


def triplet_loss(_, similarity_matrix):
    pass


# https://github.com/strongio/keras-bert
class BertLayer(Layer):
    def __init__(
        self,
        n_fine_tune_layers=10,
        pooling="mean",
        **kwargs,
    ):
        self.n_fine_tune_layers = n_fine_tune_layers
        self.trainable = True
        self.output_size = 128  # 768
        self.pooling = pooling
        # self.bert_path = "https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/3"
        self.bert_path = "https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-8_H-128_A-2/1"
        if self.pooling not in ["first", "mean"]:
            raise NameError(
                f"Undefined pooling type (must be either first or mean, but is {self.pooling}"
            )

        super(BertLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.bert = hub.KerasLayer(self.bert_path, trainable=self.trainable)

        # Remove unused layers
        trainable_vars = self.bert.variables
        if self.pooling == "first":
            trainable_vars = [var for var in trainable_vars if "/cls/" not in var.name]
            trainable_layers = ["pooler/dense"]

        elif self.pooling == "mean":
            trainable_vars = [
                var
                for var in trainable_vars
                if "/cls/" not in var.name and "/pooler/" not in var.name
            ]

            trainable_layers = []
        else:
            raise NameError(
                f"Undefined pooling type (must be either first or mean, but is {self.pooling}"
            )

        # Select how many layers to fine tune
        for i in range(self.n_fine_tune_layers):
            trainable_layers.append(f"encoder/layer_{str(11 - i)}")

        # Update trainable vars to contain only the specified layers
        trainable_vars = [
            var
            for var in trainable_vars
            if any(l in var.name for l in trainable_layers)
        ]


        # Add to trainable weights
        for var in trainable_vars:
            self._trainable_weights.append(var)

        for var in self.bert.variables:
            if var not in self._trainable_weights:
                self._non_trainable_weights.append(var)

        super(BertLayer, self).build(input_shape)

    def call(self, inputs):
        inputs = [backend.cast(x, dtype="int32") for x in inputs]
        input_ids, input_mask, input_segments = inputs
        bert_inputs = dict(
            input_word_ids=input_ids, input_mask=input_mask, input_type_ids=input_segments
        )
        if self.pooling == "first":
            pooled = self.bert(inputs=bert_inputs)["pooled_output"]
        elif self.pooling == "mean":
            result = self.bert(inputs=bert_inputs)["sequence_output"]

            mul_mask = lambda x, m: x * tf.expand_dims(m, axis=-1)
            masked_reduce_mean = lambda x, m: tf.reduce_sum(mul_mask(x, m), axis=1) / (
                    tf.reduce_sum(m, axis=1, keepdims=True) + 1e-10)
            input_mask = tf.cast(input_mask, tf.float32)
            pooled = masked_reduce_mean(result, input_mask)
        else:
            raise NameError(f"Undefined pooling type (must be either first or mean, but is {self.pooling}")

        return pooled

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_size)


"""
def make_self_attention_encoder() -> tf.Tensor:
    with tf.variable_scope("self_attention_encoder"):
        encoder_hypers = {'self_attention_activation': 'gelu',
                          'self_attention_hidden_size': 128,
                          'self_attention_intermediate_size': 512,
                          'self_attention_num_layers': 3,
                          'self_attention_num_heads': 8,
                          'self_attention_pool_mode': 'weighted_mean'}
        hypers = {'token_vocab_size': 10000,
                  'token_embedding_size': 128,
                  'use_subtokens': False,
                  'mark_subtoken_end': False,
                  'max_num_tokens': 200,
                  'use_bpe': True,
                  'pct_bpe': 0.5}
        hypers.update(encoder_hypers)

        config = BertConfig(vocab_size=10000,
                            hidden_size=128,
                            num_hidden_layers=3,
                            num_attention_heads=8,
                            intermediate_size=512)

        model = BertModel(config=config,
                          is_training=is_train,
                          input_ids=self.placeholders['tokens'],
                          input_mask=self.placeholders['tokens_mask'],
                          use_one_hot_embeddings=False)

        output_pool_mode = 'weighted_mean'
        if output_pool_mode == 'bert':
            return model.get_pooled_output()
        else:
            seq_token_embeddings = model.get_sequence_output()
            seq_token_masks = self.placeholders['tokens_mask']
            seq_token_lengths = tf.reduce_sum(seq_token_masks, axis=1)  # B
            return pool_sequence_embedding(output_pool_mode,
                                           sequence_token_embeddings=seq_token_embeddings,
                                           sequence_lengths=seq_token_lengths,
                                           sequence_token_masks=seq_token_masks)
"""


def get_bert_embedding_layer(data_type: str):
    seq_length = utils.get_seq_length(data_type)
    input_ids = Input(shape=(seq_length,), name=f'{data_type}_input_ids')
    input_masks = Input(shape=(seq_length,), name=f'{data_type}_input_masks')
    input_segments = Input(shape=(seq_length,), name=f'{data_type}_input_segments')
    bert_inputs = [input_ids, input_masks, input_segments]

    bert_output = BertLayer(n_fine_tune_layers=3, name=f'{data_type}_embedding_mean')(bert_inputs)
    # dense = Dense(shared.EMBEDDING_SIZE, activation='relu')(bert_output)
    # pred = Dense(1, activation="sigmoid")(dense)

    # model = Model(inputs=bert_inputs, outputs=pred)
    # model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
    # model.summary()
    return bert_inputs, bert_output


def get_bert2_embedding_layer(data_type: str):
    bert_layer = BertModelLayer(**BertModelLayer.Params(
        num_heads=8,
        num_layers=3,  # transformer encoder params
        vocab_size=10000,  # embedding params
        token_type_vocab_size=16,
        hidden_size=128,
        # hidden_dropout=0.1,
        intermediate_size=4 * 128,
        intermediate_activation='gelu',
        adapter_size=None,  # see arXiv:1902.00751 (adapter-BERT)
        shared_layer=False,  # True for ALBERT (arXiv:1909.11942)
        embedding_size=None,  # None for BERT, wordpiece embedding size for ALBERT
    ))

    seq_length = utils.get_seq_length(data_type)
    inputs = Input(shape=(seq_length,), name=f'{data_type}_input')
    outputs = bert_layer(inputs)  # output: [batch_size, max_seq_len, hidden_size]
    # outputs = Dense(shared.EMBEDDING_SIZE, activation='relu')(outputs)
    # weights = Dense(1, activation='sigmoid')(outputs)
    # print(outputs.get_shape().as_list())
    # print(weights.get_shape().as_list())
    # outputs = outputs * weights
    # print(outputs.get_shape().as_list())
    outputs = Lambda(mask_aware_mean, mask_aware_shape, name=f'{data_type}_embedding_mean')(outputs)
    # model = Model(inputs=input_ids, outputs=output)
    # model.build(input_shape=(None, max_seq_len))
    return inputs, outputs


def get_nbow_embedding_layer(data_type: str):
    seq_length = utils.get_seq_length(data_type)
    inputs = Input(shape=(seq_length,), name=f'{data_type}_input')
    embeddings = Embedding(
        input_length=seq_length,
        input_dim=shared.VOCAB_SIZE,
        output_dim=shared.EMBEDDING_SIZE,
        name=f'{data_type}_embedding',
        mask_zero=True)(inputs)
    # embeddings = Dropout(0.5)(embeddings)
    if shared.ATTENTION:
        embeddings = SelfAttention(16, shared.EMBEDDING_SIZE)(embeddings)
    embeddings = ZeroMaskedEntries()(embeddings)
    embeddings = Lambda(mask_aware_mean, mask_aware_shape, name=f'{data_type}_embedding_mean')(embeddings)
    return inputs, embeddings


def get_embedding_predictor(model, data_type: str):
    if shared.CONTEXT and shared.BERT1:
        input_ids = model.get_layer(f'{data_type}_input_ids').input
        input_masks = model.get_layer(f'{data_type}_input_masks').input
        input_segments = model.get_layer(f'{data_type}_input_segments').input
        inputs = [input_ids, input_masks, input_segments]
    else:
        inputs = model.get_layer(f'{data_type}_input').input
    outputs = model.get_layer(f'{data_type}_embedding_mean').output
    return Model(inputs=inputs, outputs=outputs)


def get_model() -> Model:
    inputs = list()
    embeddings = []
    for data_type in shared.SUB_TYPES:
        if shared.CONTEXT:
            if shared.BERT1:
                input_data, embedding_data = get_bert_embedding_layer(data_type)
            else:
                input_data, embedding_data = get_bert2_embedding_layer(data_type)
        else:
            input_data, embedding_data = get_nbow_embedding_layer(data_type)
        inputs.append(input_data)
        embeddings.append(embedding_data)
    embeddings = utils.repack_embeddings(embeddings)

    if shared.CONTEXT:
        merge_layer = Lambda(softmax_similarity)(embeddings)
        model = Model(inputs=inputs, outputs=merge_layer)
        model.compile(optimizer=Adam(learning_rate=shared.LEARNING_RATE), loss=softmax_loss)
    else:
        merge_layer = Lambda(cosine_similarity)(embeddings)
        model = Model(inputs=inputs, outputs=merge_layer)
        model.compile(optimizer=Adam(learning_rate=shared.LEARNING_RATE), loss=cosine_loss)
    return model


def generate_batch(train_seqs_dict, batch_size: int):
    n_samples = list(train_seqs_dict.values())[0].shape[0]

    shuffled_indices = np.arange(0, n_samples)
    np.random.shuffle(shuffled_indices)
    for data_type in train_seqs_dict.keys():
        train_seqs = train_seqs_dict.get(data_type)
        train_seqs = train_seqs[shuffled_indices, :]
        train_seqs_dict[data_type] = train_seqs

    idx = 0
    while True:
        end_idx = min(idx + batch_size, n_samples)
        n_batch_samples = min(batch_size, end_idx - idx)

        batch_seqs_dict = {}
        for data_type, encoded_seqs in train_seqs_dict.items():
            batch_encoded_seqs = encoded_seqs[idx:end_idx, :]
            if shared.CONTEXT and shared.BERT1:
                input_ids = batch_encoded_seqs
                # src/utils/tfutils.py@32
                # input_masks = np.where(input_ids > 0, 1, 0)
                input_masks = np.where(input_ids == shared.encoded_pads_dict[data_type], 0, 1)
                input_segments = np.zeros_like(input_ids)
                batch_seqs_dict[f'{data_type}_input_ids'] = input_ids
                batch_seqs_dict[f'{data_type}_input_masks'] = input_masks
                batch_seqs_dict[f'{data_type}_input_segments'] = input_segments
            else:
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


def mask_aware_shape(input_shape):
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
    for language in shared.LANGUAGES:
        train_seqs_dict = dict()
        valid_seqs_dict = {}
        for data_type in shared.SUB_TYPES:
            # train_seqs
            train_seqs = utils.load_seq(language, 'train', data_type)
            train_seqs_dict[data_type] = train_seqs
            # valid_seqs
            valid_seqs = utils.load_seq(language, 'valid', data_type)
            valid_seqs_dict[data_type] = valid_seqs
            # encoded_pads
            vocabs = utils.load_vocab(language, data_type)
            shared.encoded_pads_dict[data_type] = vocabs.word_vocab[vocabs.PAD]
        # Check for invalid sequences when it is not for evaluation
        train_seqs_dict = utils.filter_valid_seqs(train_seqs_dict)
        valid_seqs_dict = utils.filter_valid_seqs(valid_seqs_dict)
        train_model(language, train_seqs_dict, valid_seqs_dict)
