import pickle
from pathlib import Path
from typing import Dict, Any

import tensorflow as tf
from utils.tfutils import write_to_feed_dict, pool_sequence_embedding

from collections import Counter
import numpy as np
from typing import Dict, Any, List, Iterable, Optional, Tuple
import random
import re

from utils.bpevocabulary import BpeVocabulary
from utils.tfutils import convert_and_pad_token_sequence

import tensorflow as tf
from dpu_utils.codeutils import split_identifier_into_parts
from dpu_utils.mlutils import Vocabulary

from .seq_encoder import SeqEncoder


class AlonEncoder(SeqEncoder):
    @classmethod
    def get_default_hyperparameters(cls) -> Dict[str, Any]:
        encoder_hypers = {'rnn_hidden_dim': 64,
                          'rnn_dropout_keep_rate': 0.8,  # 0.5
                          'alon_pool_mode': 'alon_final',  # weighted_mean
                          }
        hypers = super().get_default_hyperparameters()
        hypers.update(encoder_hypers)
        return hypers

    @classmethod
    def _to_subtoken_stream(cls, input_stream: Iterable[str], mark_subtoken_end: bool) -> Iterable[str]:
        for token in input_stream:
            if SeqEncoder.IDENTIFIER_TOKEN_REGEX.match(token):
                yield from split_identifier_into_parts(token)
                if mark_subtoken_end:
                    yield '</id>'
            else:
                yield token

    @staticmethod
    def load_data(path, lang):
        # todo replace the existing legacy code
        contexts_file = path / f'{lang}_contexts.csv'
        ast_contexts = list()
        with open(contexts_file, 'r') as file:
            context_lines = file.readlines()
            for context_line in context_lines:
                ast_paths = context_line.split()
                ast_contexts.append(ast_paths)
            print(f'ast_contexts loaded from: {contexts_file}')
        stats_file = path / f'{lang}_stats.pkl'
        with open(stats_file, 'rb') as file:
            terminals_stats = pickle.load(file)
            nonterminals_stats = pickle.load(file)
            print(f'stats data loaded from: {stats_file}')
        return ast_paths, terminals_stats, nonterminals_stats

    @classmethod
    def brew_metadata_from_sample(cls, data_to_load: Iterable[str], raw_metadata: Dict[str, Any], lang) -> None:
        # if use_subtokens:
        #     data_to_load = cls._to_subtoken_stream(data_to_load, mark_subtoken_end=mark_subtoken_end)
        # todo check
        # load preprocessed-data
        path = Path(f'C:\\Users\\jian\\Documents\\Corpus\\{lang}\\{lang}\\final\\jsonl')
        ast_paths, terminals_stats, nonterminals_stats = cls.load_data(path, lang)
        raw_metadata['token_counter'].update(data_to_load)

    @classmethod
    def finalise_metadata(cls, encoder_label: str, hyperparameters: Dict[str, Any], raw_metadata_list: List[Dict[str, Any]]) -> Dict[str, Any]:
        # update vocabulary todo
        final_metadata = super().finalise_metadata(encoder_label, hyperparameters, raw_metadata_list)
        merged_token_counter = Counter()
        for raw_metadata in raw_metadata_list:
            merged_token_counter += raw_metadata['token_counter']

        if hyperparameters['%s_use_bpe' % encoder_label]:
            token_vocabulary = BpeVocabulary(vocab_size=hyperparameters['%s_token_vocab_size' % encoder_label],
                                             pct_bpe=hyperparameters['%s_pct_bpe' % encoder_label]
                                             )
            token_vocabulary.fit(merged_token_counter)
        else:
            token_vocabulary = Vocabulary.create_vocabulary(tokens=merged_token_counter,
                                                            max_size=hyperparameters['%s_token_vocab_size' % encoder_label],
                                                            count_threshold=hyperparameters['%s_token_vocab_count_threshold' % encoder_label])

        final_metadata['token_vocab'] = token_vocabulary
        # Save the most common tokens for use in data augmentation:
        final_metadata['common_tokens'] = merged_token_counter.most_common(50)
        return final_metadata

    def __init__(self, label: str, hyperparameters: Dict[str, Any], metadata: Dict[str, Any]):
        super().__init__(label, hyperparameters, metadata)

    @property
    def output_representation_size(self):
        return 2 * self.get_hyper('rnn_hidden_dim')

    def make_model(self, is_train: bool = False) -> tf.Tensor:
        with tf.variable_scope("alon_encoder"):
            self._make_placeholders()

            model = AlonModel()

            self.placeholders['tokens_lengths'] = \
                tf.placeholder(tf.int32, shape=[None], name='tokens_lengths')

            self.placeholders['rnn_dropout_keep_rate'] = \
                tf.placeholder(tf.float32, shape=[], name='rnn_dropout_keep_rate')

            return model.make_model(is_train)

    def init_minibatch(self, batch_data: Dict[str, Any]) -> None:
        super().init_minibatch(batch_data)
        batch_data['tokens'] = []
        batch_data['tokens_lengths'] = []

    def minibatch_to_feed_dict(self, batch_data: Dict[str, Any], feed_dict: Dict[tf.Tensor, Any],
                               is_train: bool) -> None:
        super().minibatch_to_feed_dict(batch_data, feed_dict, is_train)
        feed_dict[self.placeholders['rnn_dropout_keep_rate']] = \
            self.get_hyper('rnn_dropout_keep_rate') if is_train else 1.0

        write_to_feed_dict(feed_dict, self.placeholders['tokens'], batch_data['tokens'])
        write_to_feed_dict(feed_dict, self.placeholders['tokens_lengths'], batch_data['tokens_lengths'])


PAD = '<PAD>'
UNK = '<UNK>'


class AlonModel:
    def __init__(self):
        # config
        self.MAX_CONTEXTS = 200
        self.EMBED_SIZE = 128
        self.RNN_SIZE = 128 * 2  # Two LSTMs to embed paths, each of size 128
        self.MAX_PATH_LENGTH = 9
        self.MAX_NAME_PARTS = 5
        self.EMBED_DROPOUT_KEEP_PROB = 0.75
        self.RNN_DROPOUT_KEEP_PROB = 0.5
        # The context vector is actually a concatenation of the embedded
        # leaf & token vectors and the embedded path vector.
        self.CONTEXT_VECTOR_SIZE = 3 * self.EMBED_SIZE
        self.CODE_VECTOR_SIZE = self.EMBED_SIZE

        with open('stats.c2s', 'rb') as file:
            leaf_stats = pickle.load(file)
            node_stats = pickle.load(file)
            print('stats loaded.')

        leaf2index, self.leaf_size = self.stats2data(leaf_stats)
        print('Loaded leaf token size: %d' % self.leaf_size)
        node2index, self.node_size = self.stats2data(node_stats)
        print('Loaded node token size: %d' % self.node_size)

        self.leaf_table = self.initialize_table(leaf2index, leaf2index[UNK])
        self.node_table = self.initialize_table(node2index, node2index[UNK])
        self.context_pad = f'{PAD},{PAD},{PAD}'

        record_defaults = [[self.context_pad]] * self.MAX_CONTEXTS
        dataset = tf.data.experimental.CsvDataset(
            'context.c2s', record_defaults=record_defaults, field_delim=' ',
            use_quote_delim=False, buffer_size=100 * 1024 * 1024)
        # dataset = dataset.shuffle(10000, reshuffle_each_iteration=True)
        dataset = dataset.apply(tf.data.experimental.map_and_batch(
            map_func=self.process_dataset, batch_size=256))
        dataset = dataset.prefetch(tf.contrib.data.AUTOTUNE)
        iterator = dataset.make_initializable_iterator()
        self.input_tensors = iterator.get_next()

    @staticmethod
    def stats2data(stats):
        token2index = {}
        size = 0
        for value in [PAD, UNK]:
            token2index[value] = size
            size += 1
        sorted_counts = [(k, stats[k]) for k in sorted(stats, key=stats.get, reverse=True)]
        limited_sorted = dict(sorted_counts)
        for token, _ in limited_sorted.items():
            token2index[token] = size
            size += 1
        return token2index, size

    @staticmethod
    def initialize_table(token2index, default_value):
        return tf.contrib.lookup.HashTable(
            tf.contrib.lookup.KeyValueTensorInitializer(list(token2index.keys()), list(token2index.values()),
                                                        key_dtype=tf.string, value_dtype=tf.int32), default_value)

    def context2info(self, context, index):
        # (max_contexts, 1)
        context_str = tf.slice(context, [0, index], [self.MAX_CONTEXTS, 1])
        # (max_contexts)
        flat_context_str = tf.reshape(context_str, [-1])
        # (max_contexts, max_name_parts)
        context_tokens = tf.string_split(flat_context_str, delimiter='|', skip_empty=False)
        if index in (0, 2):
            max_length = tf.maximum(tf.to_int64(self.MAX_NAME_PARTS), context_tokens.dense_shape[1])
        else:
            max_length = self.MAX_PATH_LENGTH
        # (batch, max_contexts, max_length)
        # (max_contexts, max_length)
        sparse_tokens = tf.sparse.SparseTensor(indices=context_tokens.indices, values=context_tokens.values,
                                               dense_shape=[self.MAX_CONTEXTS, max_length])
        dense_tokens = tf.sparse.to_dense(sp_input=sparse_tokens, default_value=PAD)
        if index in (0, 2):
            dense_tokens = tf.slice(dense_tokens, [0, 0], [-1, self.MAX_NAME_PARTS])
            # (max_contexts, max_length)
            token_ids = self.leaf_table.lookup(dense_tokens)
        else:
            # (max_contexts, max_length)
            token_ids = self.node_table.lookup(dense_tokens)
        # (max_contexts)
        token_lengths = tf.reduce_sum(tf.cast(tf.not_equal(dense_tokens, PAD), tf.int32), -1)
        return token_ids, token_lengths

    def process_dataset(self, *row_parts):
        row_parts = list(row_parts)

        # all_contexts = tf.stack(row_parts)
        # all_contexts_padded = tf.concat([all_contexts, [self.context_pad]], axis=-1)
        # index_of_blank_context = tf.where(tf.equal(all_contexts_padded, self.context_pad))
        # num_contexts_per_example = tf.reduce_min(index_of_blank_context)
        # # if there are less than self.max_contexts valid contexts, still sample self.max_contexts
        # safe_limit = tf.cast(tf.maximum(num_contexts_per_example, self.MAX_CONTEXTS), tf.int32)
        # rand_indices = tf.random_shuffle(tf.range(safe_limit))[:self.MAX_CONTEXTS]
        # context_str = tf.gather(all_contexts, rand_indices)  # (max_contexts,)
        context_str = row_parts[:self.MAX_CONTEXTS]  # (max_contexts,)
        ######
        # contexts (max_contexts, )
        contexts = tf.string_split(context_str, delimiter=',', skip_empty=False)
        shape4contexts = [self.MAX_CONTEXTS, 3]
        sparse_contexts = tf.sparse.SparseTensor(indices=contexts.indices,
                                                 values=contexts.values,
                                                 dense_shape=shape4contexts)
        dense_contexts = tf.sparse.to_dense(sp_input=sparse_contexts, default_value=PAD)
        # (batch, max_contexts, 3)
        dense_contexts = tf.reshape(dense_contexts, shape=shape4contexts)
        #########
        source_ids, source_lengths = self.context2info(dense_contexts, 0)
        middle_ids, middle_lengths = self.context2info(dense_contexts, 1)
        target_ids, target_lengths = self.context2info(dense_contexts, 2)
        #########
        context_mask = tf.to_float(tf.not_equal(
            tf.reduce_max(source_ids, -1) + tf.reduce_max(middle_ids, -1) + tf.reduce_max(target_ids, -1), 0))

        return {'CONTEXT_MASK': context_mask, 'SOURCE_IDS': source_ids,
                'MIDDLE_IDS': middle_ids, 'TARGET_IDS': target_ids,
                'SOURCE_LENGTHS': source_lengths, 'MIDDLE_LENGTHS': middle_lengths,
                'TARGET_LENGTHS': target_lengths}

    def make_model(self, is_train: bool = False):
        context_mask = self.input_tensors['CONTEXT_MASK']
        source_ids = self.input_tensors['SOURCE_IDS']
        middle_ids = self.input_tensors['MIDDLE_IDS']
        target_ids = self.input_tensors['TARGET_IDS']
        source_lengths = self.input_tensors['SOURCE_LENGTHS']
        middle_lengths = self.input_tensors['MIDDLE_LENGTHS']
        target_lengths = self.input_tensors['TARGET_LENGTHS']

        with tf.variable_scope('alon'):
            leaf_token = tf.get_variable(
                'LEAF_TOKEN', shape=(self.leaf_size, self.EMBED_SIZE), dtype=tf.float32,
                initializer=tf.contrib.layers.variance_scaling_initializer(factor=1.0, mode='FAN_OUT', uniform=True))
            node_token = tf.get_variable(
                'NODE_TOKEN', shape=(self.node_size, self.EMBED_SIZE), dtype=tf.float32,
                initializer=tf.contrib.layers.variance_scaling_initializer(factor=1.0, mode='FAN_OUT', uniform=True))

            # (batch, max_contexts, max_name_parts, dim)
            source_embed = tf.nn.embedding_lookup(params=leaf_token, ids=source_ids)
            # (batch, max_contexts, max_path_length+1, dim)
            middle_embed = tf.nn.embedding_lookup(params=node_token, ids=middle_ids)
            # (batch, max_contexts, max_name_parts, dim)
            target_embed = tf.nn.embedding_lookup(params=leaf_token, ids=target_ids)

            # (batch, max_contexts, max_name_parts, 1)
            source_mask = tf.expand_dims(
                tf.sequence_mask(source_lengths, maxlen=self.MAX_NAME_PARTS, dtype=tf.float32), -1)
            # (batch, max_contexts, max_name_parts, 1)
            target_mask = tf.expand_dims(
                tf.sequence_mask(target_lengths, maxlen=self.MAX_NAME_PARTS, dtype=tf.float32), -1)

            # (batch, max_contexts, dim)
            source_token_sum = tf.reduce_sum(source_embed * source_mask, axis=2)
            # (batch, max_contexts, rnn_size)
            middle_aggregation = self.aggregate_path(middle_embed, middle_lengths, context_mask, is_train)
            # (batch, max_contexts, dim)
            target_token_sum = tf.reduce_sum(target_embed * target_mask, axis=2)

            # (batch, max_contexts, dim * 2 + rnn_size)
            context_embed = tf.concat([source_token_sum, middle_aggregation, target_token_sum], axis=-1)
            # (batch, max_contexts, dim * 3)
            # context_embed = tf.concat([source_token_sum, middle_aggregation, target_token_sum], axis=-1)
            if is_train:
                context_embed = tf.nn.dropout(context_embed, self.EMBED_DROPOUT_KEEP_PROB)

            # (batch * max_contexts, dim * 3)
            flat_embed = tf.reshape(context_embed, [-1, self.CONTEXT_VECTOR_SIZE])
            transform_shape = (self.CONTEXT_VECTOR_SIZE, self.CODE_VECTOR_SIZE)
            transform_param = tf.get_variable('TRANSFORM', shape=transform_shape, dtype=tf.float32)

            # (batch * max_contexts, dim * 3)
            flat_embed = tf.tanh(tf.matmul(flat_embed, transform_param))
            attention_shape = (self.CODE_VECTOR_SIZE, 1)
            attention_param = tf.get_variable('ATTENTION', shape=attention_shape, dtype=tf.float32)

            # (batch * max_contexts, 1)
            contexts_weights = tf.matmul(flat_embed, attention_param)
            # (batch, max_contexts, 1)
            batched_contexts_weights = tf.reshape(contexts_weights, [-1, self.MAX_CONTEXTS, 1])
            # (batch, max_contexts)
            mask = tf.math.log(context_mask)
            # (batch, max_contexts, 1)
            mask = tf.expand_dims(mask, axis=2)
            # (batch, max_contexts, 1)
            batched_contexts_weights += mask
            # (batch, max_contexts, 1)
            attention_weights = tf.nn.softmax(batched_contexts_weights, axis=1)

            batched_shape = (-1, self.MAX_CONTEXTS, self.CODE_VECTOR_SIZE)
            batched_embed = tf.reshape(flat_embed, shape=batched_shape)
            # (batch, dim * 3)
            # (batch, max_contexts, vector_size)
            code_vectors = tf.reduce_sum(tf.multiply(batched_embed, attention_weights), axis=1)

        return code_vectors

    def aggregate_path(self, path_embed, path_lengths, context_mask, is_train=False):
        # path_embed:           (batch, max_contexts, max_path_length+1, dim)
        # path_length:          (batch, max_contexts)
        # context_mask:         (batch, max_contexts)
        max_contexts = tf.shape(path_embed)[1]
        # (batch * max_contexts, max_path_length+1, dim)
        flat_paths = tf.reshape(path_embed, shape=[-1, self.MAX_PATH_LENGTH, self.EMBED_SIZE])
        # (batch * max_contexts)
        flat_context_mask = tf.reshape(context_mask, [-1])
        # (batch * max_contexts)
        lengths = tf.multiply(tf.reshape(path_lengths, [-1]), tf.cast(flat_context_mask, tf.int32))

        # BiRNN:
        rnn_cell_fw = tf.nn.rnn_cell.LSTMCell(self.RNN_SIZE // 2)
        rnn_cell_bw = tf.nn.rnn_cell.LSTMCell(self.RNN_SIZE // 2)
        if is_train:
            rnn_cell_fw = tf.nn.rnn_cell.DropoutWrapper(rnn_cell_fw, output_keep_prob=self.RNN_DROPOUT_KEEP_PROB)
            rnn_cell_bw = tf.nn.rnn_cell.DropoutWrapper(rnn_cell_bw, output_keep_prob=self.RNN_DROPOUT_KEEP_PROB)

        # Merge fwd/bwd outputs:
        # _, final_states = tf.nn.bidirectional_dynamic_rnn(
        #     cell_fw=rnn_cell_fw, cell_bw=rnn_cell_bw, inputs=flat_paths, sequence_length=lengths, dtype=tf.float32)
        # rnn_final_state = tf.concat([tf.concat([tf.concat(layer_final_state, axis=-1)  # concat c & m of LSTM cell
        #                                         for layer_final_state in layer_final_states],
        #                                        axis=-1)  # concat across layers
        #                              for layer_final_states in final_states],
        #                             axis=-1)  # concat fwd & bwd
        _, (state_fw, state_bw) = tf.nn.bidirectional_dynamic_rnn(
            cell_fw=rnn_cell_fw, cell_bw=rnn_cell_bw, inputs=flat_paths, sequence_length=lengths, dtype=tf.float32)
        # (batch * max_contexts, rnn_size)
        rnn_final_state = tf.concat([state_fw.h, state_bw.h], axis=-1)

        # (batch, max_contexts, rnn_size)
        return tf.reshape(rnn_final_state, shape=[-1, max_contexts, self.RNN_SIZE])
