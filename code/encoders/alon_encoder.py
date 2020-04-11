from typing import Dict, Any, List, Iterable, Optional, Tuple

import tensorflow as tf
from utils.tfutils import write_to_feed_dict

from collections import Counter
import numpy as np
import random

from utils.tfutils import convert_and_pad_token_sequence

from dpu_utils.codeutils import split_identifier_into_parts
from dpu_utils.mlutils import Vocabulary

from .seq_encoder import Encoder, QueryType
from scripts.ts import code2paths
from scripts.run import get_path, load_data, paths2tokens


class AlonEncoder(Encoder):
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
    def init_metadata(cls) -> Dict[str, Any]:
        raw_metadata = super().init_metadata()
        raw_metadata['ast_contexts'] = list()
        raw_metadata['context_filenames'] = list()
        raw_metadata['terminal_counter'] = Counter()
        raw_metadata['nonterminal_counter'] = Counter()
        return raw_metadata

    @classmethod
    def load_metadata_from_sample(cls, language: str, data_to_brew: Iterable[str], data_to_load: Iterable[str],
                                  raw_metadata: Dict[str, Any], use_subtokens: bool=False, mark_subtoken_end: bool=False) -> None:
        # JGD load preprocessed data
        path = get_path(language)
        print('$A1C1', end='\r')
        context_filename, terminal_counter, nonterminal_counter = load_data(path)
        print('$A1C2', end='\r')
        raw_metadata['context_filenames'].extend(context_filename)
        print('$A1C3', end='\r')
        raw_metadata['terminal_counter'] += terminal_counter
        print('$A1C4', end='\r')
        raw_metadata['nonterminal_counter'] += nonterminal_counter
        print('$A1C5', end='\r')

    @classmethod
    def finalise_metadata(cls, encoder_label: str, hyperparameters: Dict[str, Any], raw_metadata_list: List[Dict[str, Any]]) -> Dict[str, Any]:
        # JGD build vocabulary ?
        final_metadata = super().finalise_metadata(encoder_label, hyperparameters, raw_metadata_list)
        merged_context_filenames = list()
        merged_terminal_counter = Counter()
        merged_nonterminal_counter = Counter()
        for raw_metadata in raw_metadata_list:
            merged_context_filenames.extend(raw_metadata['context_filenames'])
            merged_terminal_counter += raw_metadata['terminal_counter']
            merged_nonterminal_counter += raw_metadata['nonterminal_counter']

        final_metadata['context_filenames'] = merged_context_filenames
        final_metadata['terminal_counter'] = merged_terminal_counter
        final_metadata['nonterminal_counter'] = merged_nonterminal_counter

        # if hyperparameters['%s_use_bpe' % encoder_label]:
        #     token_vocabulary = BpeVocabulary(vocab_size=hyperparameters['%s_token_vocab_size' % encoder_label],
        #                                      pct_bpe=hyperparameters['%s_pct_bpe' % encoder_label]
        #                                      )
        #     token_vocabulary.fit(merged_token_counter)
        # else:
        #     token_vocabulary = Vocabulary.create_vocabulary(tokens=merged_token_counter,
        #                                                     max_size=hyperparameters['%s_token_vocab_size' % encoder_label],
        #                                                     count_threshold=hyperparameters['%s_token_vocab_count_threshold' % encoder_label])
        #
        # final_metadata['token_vocab'] = token_vocabulary
        # # Save the most common tokens for use in data augmentation:
        # final_metadata['common_tokens'] = merged_token_counter.most_common(50)
        return final_metadata

    def __init__(self, label: str, hyperparameters: Dict[str, Any], metadata: Dict[str, Any]):
        super().__init__(label, hyperparameters, metadata)

    @classmethod
    def load_data_from_sample(cls,
                              encoder_label: str,
                              hyperparameters: Dict[str, Any],
                              metadata: Dict[str, Any],
                              language: str,
                              data_to_brew: Any,
                              data_to_load: Any,
                              function_name: Optional[str],
                              result_holder: Dict[str, Any],
                              is_test: bool = True) -> bool:
        """
        Saves two versions of both the code and the query: one using the docstring as the query and the other using the
        function-name as the query, and replacing the function name in the code with an out-of-vocab token.
        Sub-tokenizes, converts, and pads both versions, and rejects empty samples.
        """
        # JGD extract AST-paths
        ast_paths = code2paths(data_to_brew, language)  # JGD check ...
        terminals, nonterminals = paths2tokens(ast_paths)
        # count the tokens
        terminal_counter = Counter(terminals)
        nonterminal_counter = Counter(nonterminals)
        # JGD todo check
        result_holder['ast_contexts'] = ast_paths
        result_holder['terminal_counter'] = terminal_counter
        result_holder['nonterminal_counter'] = nonterminal_counter

        # # Save the two versions of the code and query:
        # data_holder = {QueryType.DOCSTRING.value: data_to_load, QueryType.FUNCTION_NAME.value: None}
        # # Skip samples where the function name is very short, because it probably has too little information
        # # to be a good search query.
        # if not is_test and hyperparameters['fraction_using_func_name'] > 0. and function_name and \
        #         len(function_name) >= hyperparameters['min_len_func_name_for_query']:
        #     if encoder_label == 'query':
        #         print('NOT EXPECTED')
        #         # Set the query tokens to the function name, broken up into its sub-tokens:
        #         data_holder[QueryType.FUNCTION_NAME.value] = split_identifier_into_parts(function_name)
        #     elif encoder_label == 'code':
        #         # In the code, replace the function name with the out-of-vocab token everywhere it appears:
        #         data_holder[QueryType.FUNCTION_NAME.value] = [Vocabulary.get_unk() if token == function_name else token
        #                                                       for token in data_to_load]
        #
        # # Sub-tokenize, convert, and pad both versions:
        # for key, data in data_holder.items():
        #     if not data:
        #         result_holder[f'{encoder_label}_tokens_{key}'] = None
        #         result_holder[f'{encoder_label}_tokens_mask_{key}'] = None
        #         result_holder[f'{encoder_label}_tokens_length_{key}'] = None
        #         continue
        #     if hyperparameters[f'{encoder_label}_use_subtokens']:
        #         data = cls._to_subtoken_stream(data,
        #                                        mark_subtoken_end=hyperparameters[
        #                                            f'{encoder_label}_mark_subtoken_end'])
        #     tokens, tokens_mask = \
        #         convert_and_pad_token_sequence(metadata['token_vocab'], list(data),
        #                                        hyperparameters[f'{encoder_label}_max_num_tokens'])
        #     # Note that we share the result_holder with different encoders, and so we need to make our identifiers
        #     # unique-ish
        #     result_holder[f'{encoder_label}_tokens_{key}'] = tokens
        #     result_holder[f'{encoder_label}_tokens_mask_{key}'] = tokens_mask
        #     result_holder[f'{encoder_label}_tokens_length_{key}'] = int(np.sum(tokens_mask))
        #
        # if result_holder[f'{encoder_label}_tokens_mask_{QueryType.DOCSTRING.value}'] is None or \
        #         int(np.sum(result_holder[f'{encoder_label}_tokens_mask_{QueryType.DOCSTRING.value}'])) == 0:
        #     return False

        return True

    def extend_minibatch_by_sample(self, batch_data: Dict[str, Any], sample: Dict[str, Any], is_train: bool=False,
                                   query_type: QueryType = QueryType.DOCSTRING.value) -> bool:
        """
        Implements various forms of data augmentation.
        """
        # current_sample = dict()
        #
        # # Train with some fraction of samples having their query set to the function name instead of the docstring, and
        # # their function name replaced with out-of-vocab in the code:
        # current_sample['tokens'] = sample[f'{self.label}_tokens_{query_type}']
        # current_sample['tokens_mask'] = sample[f'{self.label}_tokens_mask_{query_type}']
        # current_sample['tokens_lengths'] = sample[f'{self.label}_tokens_length_{query_type}']
        #
        # # In the query, randomly add high-frequency tokens:
        # # TODO: Add tokens with frequency proportional to their frequency in the vocabulary
        # if is_train and self.label == 'query' and self.hyperparameters['query_random_token_frequency'] > 0.:
        #     total_length = len(current_sample['tokens'])
        #     length_without_padding = current_sample['tokens_lengths']
        #     # Generate a list of places in which to insert tokens:
        #     insert_indices = np.array([random.uniform(0., 1.) for _ in range(length_without_padding)])  # don't allow insertions in the padding
        #     insert_indices = insert_indices < self.hyperparameters['query_random_token_frequency']  # insert at the correct frequency
        #     insert_indices = np.flatnonzero(insert_indices)
        #     if len(insert_indices) > 0:
        #         # Generate the random tokens to add:
        #         tokens_to_add = [random.randrange(0, len(self.metadata['common_tokens']))
        #                          for _ in range(len(insert_indices))]  # select one of the most common tokens for each location
        #         tokens_to_add = [self.metadata['common_tokens'][token][0] for token in tokens_to_add]  # get the word corresponding to the token we're adding
        #         tokens_to_add = [self.metadata['token_vocab'].get_id_or_unk(token) for token in tokens_to_add]  # get the index within the vocab of the token we're adding
        #         # Efficiently insert the added tokens, leaving the total length the same:
        #         to_insert = 0
        #         output_query = np.zeros(total_length, dtype=int)
        #         for idx in range(min(length_without_padding, total_length - len(insert_indices))):  # iterate only through the beginning of the array where changes are being made
        #             if to_insert < len(insert_indices) and idx == insert_indices[to_insert]:
        #                 output_query[idx + to_insert] = tokens_to_add[to_insert]
        #                 to_insert += 1
        #             output_query[idx + to_insert] = current_sample['tokens'][idx]
        #         current_sample['tokens'] = output_query
        #         # Add the needed number of non-padding values to the mask:
        #         current_sample['tokens_mask'][length_without_padding:length_without_padding + len(tokens_to_add)] = 1.
        #         current_sample['tokens_lengths'] += len(tokens_to_add)
        #
        # # Add the current sample to the minibatch:
        # [batch_data[key].append(current_sample[key]) for key in current_sample.keys() if key in batch_data.keys()]

        return False

    def embedding_layer(self, token_inp: tf.Tensor) -> tf.Tensor:
        """
        Creates embedding layer that is in common between many encoders.

        Args:
            token_inp:  2D tensor that is of shape (batch size, sequence length)

        Returns:
            3D tensor of shape (batch size, sequence length, embedding dimension)
        """

        token_embeddings = tf.get_variable(name='token_embeddings',
                                           initializer=tf.glorot_uniform_initializer(),
                                           shape=[len(self.metadata['token_vocab']),
                                                  self.get_hyper('token_embedding_size')],
                                           )
        self.__embeddings = token_embeddings

        token_embeddings = tf.nn.dropout(token_embeddings,
                                         keep_prob=self.placeholders['dropout_keep_rate'])

        return tf.nn.embedding_lookup(params=token_embeddings, ids=token_inp)

    def get_token_embeddings(self) -> Tuple[tf.Tensor, List[str]]:
        return (self.__embeddings, list(self.metadata['token_vocab'].id_to_token))

    @property
    def output_representation_size(self):
        return 2 * self.get_hyper('rnn_hidden_dim')

    def make_model(self, is_train: bool = False) -> tf.Tensor:
        with tf.variable_scope("alon_encoder"):
            self._make_placeholders()

            context_filenames = self.metadata['context_filenames']
            terminal_counter = self.metadata['terminal_counter']
            nonterminal_counter = self.metadata['nonterminal_counter']
            print('$T1******', end='\r')
            model = AlonModel(context_filenames, terminal_counter, nonterminal_counter)
            print('$T2******', end='\r')
            self.placeholders['tokens_lengths'] = \
                tf.placeholder(tf.int32, shape=[None], name='tokens_lengths')
            print('$T3******', end='\r')
            self.placeholders['rnn_dropout_keep_rate'] = \
                tf.placeholder(tf.float32, shape=[], name='rnn_dropout_keep_rate')
            print('$T4******', end='\r')
            return model.build_computation_graph(is_train)

    def init_minibatch(self, batch_data: Dict[str, Any]) -> None:
        super().init_minibatch(batch_data)
        batch_data['tokens'] = []
        batch_data['tokens_lengths'] = []

    def minibatch_to_feed_dict(self, batch_data: Dict[str, Any], feed_dict: Dict[tf.Tensor, Any],
                               is_train: bool) -> None:
        super().minibatch_to_feed_dict(batch_data, feed_dict, is_train)
        feed_dict[self.placeholders['rnn_dropout_keep_rate']] = \
            self.get_hyper('rnn_dropout_keep_rate') if is_train else 1.0

        # write_to_feed_dict(feed_dict, self.placeholders['tokens'], batch_data['tokens'])
        # write_to_feed_dict(feed_dict, self.placeholders['tokens_lengths'], batch_data['tokens_lengths'])


PAD = '<PAD>'
UNK = '<UNK>'


class AlonModel:
    def __init__(self, context_filenames, terminal_counter, nonterminal_counter):
        # config
        self.MAX_CONTEXTS = 200
        self.EMBED_SIZE = 128
        self.RNN_SIZE = 128 * 2  # Two LSTMs to embed paths, each of size 128
        self.MAX_PATH_LENGTH = 9
        self.MAX_NAME_PARTS = 5
        self.EMBED_DROPOUT_KEEP_PROB = 0.75
        self.RNN_DROPOUT_KEEP_PROB = 0.5
        # The context vector is actually a concatenation of the embedded
        # source & target vectors and the embedded middle path vector.
        self.CONTEXT_VECTOR_SIZE = 3 * self.EMBED_SIZE
        self.CODE_VECTOR_SIZE = self.EMBED_SIZE

        terminal2index, self.terminal_size = self.stats2data(terminal_counter)
        print('Loaded terminal token size: %d' % self.terminal_size)
        nonterminal2index, self.nonterminal_size = self.stats2data(nonterminal_counter)
        print('Loaded nonterminal token size: %d' % self.nonterminal_size)

        self.terminal_table = self.initialize_table(terminal2index, terminal2index[UNK])
        self.nonterminal_table = self.initialize_table(nonterminal2index, nonterminal2index[UNK])
        self.context_pad = f'{PAD},{PAD},{PAD}'

        record_defaults = [[self.context_pad]] * self.MAX_CONTEXTS
        dataset = tf.data.experimental.CsvDataset(
            context_filenames, record_defaults=record_defaults, field_delim='\n',
            use_quote_delim=False, buffer_size=1000 * 1024 * 1024)
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
        print(f'context_str\n{context_str.get_shape()}')
        # (max_contexts)
        flat_context_str = tf.reshape(context_str, [-1])
        print(f'flat_context_str\n{flat_context_str.get_shape()}')
        # (max_contexts, max_name_parts)
        context_tokens = tf.string_split(flat_context_str, delimiter='|', skip_empty=False)
        print(f'context_tokens\n{context_tokens.get_shape()}')
        if index in (0, 2):
            max_length = tf.maximum(tf.to_int64(self.MAX_NAME_PARTS), context_tokens.dense_shape[1])
            print(f'max_length\n{max_length.get_shape()}')
        else:
            max_length = tf.to_int64(self.MAX_PATH_LENGTH)
            print(f'max_length\n{max_length.get_shape()}')
        # (batch, max_contexts, max_length)
        # (max_contexts, max_length)
        sparse_tokens = tf.sparse.SparseTensor(indices=context_tokens.indices, values=context_tokens.values,
                                               dense_shape=[self.MAX_CONTEXTS, max_length])
        print(f'sparse_tokens\n{sparse_tokens.get_shape()}')
        dense_tokens = tf.sparse.to_dense(sp_input=sparse_tokens, default_value=PAD)
        print(f'dense_tokens\n{dense_tokens.get_shape()}')
        if index in (0, 2):
            dense_tokens = tf.slice(dense_tokens, [0, 0], [-1, self.MAX_NAME_PARTS])
            print(f'dense_tokens\n{dense_tokens.get_shape()}')
            # (max_contexts, max_length)
            token_ids = self.terminal_table.lookup(dense_tokens)
            print(f'token_ids\n{token_ids.get_shape()}')
        else:
            # (max_contexts, max_length)
            token_ids = self.nonterminal_table.lookup(dense_tokens)
            print(f'token_ids\n{token_ids.get_shape()}')
        # (max_contexts)
        token_lengths = tf.reduce_sum(tf.cast(tf.not_equal(dense_tokens, PAD), tf.int32), -1)
        print(f'token_lengths\n{token_lengths.get_shape()}')
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
        print(f'contexts\n{contexts.get_shape()}')
        shape4contexts = [self.MAX_CONTEXTS, 3]
        sparse_contexts = tf.sparse.SparseTensor(indices=contexts.indices,
                                                 values=contexts.values,
                                                 dense_shape=shape4contexts)
        print(f'sparse_contexts\n{sparse_contexts.get_shape()}')
        dense_contexts = tf.sparse.to_dense(sp_input=sparse_contexts, default_value=PAD)
        print(f'dense_contexts\n{dense_contexts.get_shape()}')
        # (batch, max_contexts, 3)
        dense_contexts = tf.reshape(dense_contexts, shape=shape4contexts)
        print(f'dense_contexts\n{dense_contexts.get_shape()}')
        #########
        source_ids, source_lengths = self.context2info(dense_contexts, 0)
        print(f'source_ids\n{source_ids.get_shape()}')
        print(f'source_lengths\n{source_lengths.get_shape()}')
        middle_ids, middle_lengths = self.context2info(dense_contexts, 1)
        print(f'middle_ids\n{middle_ids.get_shape()}')
        print(f'middle_lengths\n{middle_lengths.get_shape()}')
        target_ids, target_lengths = self.context2info(dense_contexts, 2)
        print(f'target_ids\n{target_ids.get_shape()}')
        print(f'target_lengths\n{target_lengths.get_shape()}')
        #########
        context_mask = tf.to_float(tf.not_equal(
            tf.reduce_max(source_ids, -1) + tf.reduce_max(middle_ids, -1) + tf.reduce_max(target_ids, -1), 0))
        print(f'context_mask\n{context_mask.get_shape()}')

        return {'CONTEXT_MASK': context_mask, 'SOURCE_IDS': source_ids,
                'MIDDLE_IDS': middle_ids, 'TARGET_IDS': target_ids,
                'SOURCE_LENGTHS': source_lengths, 'MIDDLE_LENGTHS': middle_lengths,
                'TARGET_LENGTHS': target_lengths}

    def build_computation_graph(self, is_train: bool = False):
        context_mask = self.input_tensors['CONTEXT_MASK']
        source_ids = self.input_tensors['SOURCE_IDS']
        middle_ids = self.input_tensors['MIDDLE_IDS']
        target_ids = self.input_tensors['TARGET_IDS']
        source_lengths = self.input_tensors['SOURCE_LENGTHS']
        middle_lengths = self.input_tensors['MIDDLE_LENGTHS']
        target_lengths = self.input_tensors['TARGET_LENGTHS']

        with tf.variable_scope('alon'):
            terminal_token = tf.get_variable(
                'terminal_token', shape=(self.terminal_size, self.EMBED_SIZE), dtype=tf.float32,
                initializer=tf.contrib.layers.variance_scaling_initializer(factor=1.0, mode='FAN_OUT', uniform=True))
            print(f'terminal_token\n{terminal_token.get_shape()}')
            nonterminal_token = tf.get_variable(
                'nonterminal_token', shape=(self.nonterminal_size, self.EMBED_SIZE), dtype=tf.float32,
                initializer=tf.contrib.layers.variance_scaling_initializer(factor=1.0, mode='FAN_OUT', uniform=True))
            print(f'nonterminal_token\n{nonterminal_token.get_shape()}')

            # (batch, max_contexts, max_name_parts, dim)
            source_embed = tf.nn.embedding_lookup(params=terminal_token, ids=source_ids)
            print(f'source_embed\n{source_embed.get_shape()}')
            # (batch, max_contexts, max_path_length+1, dim)
            middle_embed = tf.nn.embedding_lookup(params=nonterminal_token, ids=middle_ids)
            print(f'middle_embed\n{middle_embed.get_shape()}')
            # (batch, max_contexts, max_name_parts, dim)
            target_embed = tf.nn.embedding_lookup(params=terminal_token, ids=target_ids)
            print(f'target_embed\n{target_embed.get_shape()}')

            # (batch, max_contexts, max_name_parts, 1)
            source_mask = tf.expand_dims(
                tf.sequence_mask(source_lengths, maxlen=self.MAX_NAME_PARTS, dtype=tf.float32), -1)
            print(f'source_mask\n{source_mask.get_shape()}')
            # (batch, max_contexts, max_name_parts, 1)
            target_mask = tf.expand_dims(
                tf.sequence_mask(target_lengths, maxlen=self.MAX_NAME_PARTS, dtype=tf.float32), -1)
            print(f'target_mask\n{target_mask.get_shape()}')

            # (batch, max_contexts, dim)
            source_token_sum = tf.reduce_sum(source_embed * source_mask, axis=2)
            print(f'source_token_sum\n{source_token_sum.get_shape()}')
            # (batch, max_contexts, rnn_size)
            middle_aggregation = self.aggregate_path(middle_embed, middle_lengths, context_mask, is_train)
            print(f'middle_aggregation\n{middle_aggregation.get_shape()}')
            # (batch, max_contexts, dim)
            target_token_sum = tf.reduce_sum(target_embed * target_mask, axis=2)
            print(f'target_token_sum\n{target_token_sum.get_shape()}')

            # (batch, max_contexts, dim * 2 + rnn_size)
            context_embed = tf.concat([source_token_sum, middle_aggregation, target_token_sum], axis=-1)
            print(f'context_embed\n{context_embed.get_shape()}')
            # (batch, max_contexts, dim * 3)
            # context_embed = tf.concat([source_token_sum, middle_aggregation, target_token_sum], axis=-1)
            if is_train:
                context_embed = tf.nn.dropout(context_embed, self.EMBED_DROPOUT_KEEP_PROB)
                print(f'context_embed\n{context_embed.get_shape()}')

            # (batch * max_contexts, dim * 3)
            flat_embed = tf.reshape(context_embed, [-1, self.CONTEXT_VECTOR_SIZE])
            print(f'flat_embed\n{flat_embed.get_shape()}')
            transform_shape = (self.CONTEXT_VECTOR_SIZE, self.CODE_VECTOR_SIZE)
            transform_param = tf.get_variable('TRANSFORM', shape=transform_shape, dtype=tf.float32)
            print(f'transform_param\n{transform_param.get_shape()}')

            # (batch * max_contexts, dim * 3)
            flat_embed = tf.tanh(tf.matmul(flat_embed, transform_param))
            print(f'flat_embed\n{flat_embed.get_shape()}')
            attention_shape = (self.CODE_VECTOR_SIZE, 1)
            attention_param = tf.get_variable('ATTENTION', shape=attention_shape, dtype=tf.float32)
            print(f'attention_param\n{attention_param.get_shape()}')

            # (batch * max_contexts, 1)
            contexts_weights = tf.matmul(flat_embed, attention_param)
            print(f'contexts_weights\n{contexts_weights.get_shape()}')
            # (batch, max_contexts, 1)
            batched_contexts_weights = tf.reshape(contexts_weights, [-1, self.MAX_CONTEXTS, 1])
            print(f'batched_contexts_weights\n{batched_contexts_weights.get_shape()}')
            # (batch, max_contexts)
            mask = tf.math.log(context_mask)
            print(f'mask\n{mask.get_shape()}')
            # (batch, max_contexts, 1)
            mask = tf.expand_dims(mask, axis=2)
            print(f'mask\n{mask.get_shape()}')
            # (batch, max_contexts, 1)
            batched_contexts_weights += mask
            print(f'batched_contexts_weights\n{batched_contexts_weights.get_shape()}')
            # (batch, max_contexts, 1)
            attention_weights = tf.nn.softmax(batched_contexts_weights, axis=1)
            print(f'attention_weights\n{attention_weights.get_shape()}')

            batched_shape = (-1, self.MAX_CONTEXTS, self.CODE_VECTOR_SIZE)
            batched_embed = tf.reshape(flat_embed, shape=batched_shape)
            print(f'batched_embed\n{batched_embed.get_shape()}')
            # (batch, dim * 3)
            # (batch, max_contexts, vector_size)
            code_vectors = tf.reduce_sum(tf.multiply(batched_embed, attention_weights), axis=1)
            print(f'code_vectors\n{code_vectors.get_shape()}')

        return code_vectors

    def aggregate_path(self, path_embed, path_lengths, context_mask, is_train=False):
        # path_embed:           (batch, max_contexts, max_path_length+1, dim)
        # path_length:          (batch, max_contexts)
        # context_mask:         (batch, max_contexts)
        max_contexts = tf.shape(path_embed)[1]
        print(f'max_contexts\n{max_contexts.get_shape()}')
        # (batch * max_contexts, max_path_length+1, dim)
        flat_paths = tf.reshape(path_embed, shape=[-1, self.MAX_PATH_LENGTH, self.EMBED_SIZE])
        print(f'flat_paths\n{flat_paths.get_shape()}')
        # (batch * max_contexts)
        flat_context_mask = tf.reshape(context_mask, [-1])
        print(f'flat_context_mask\n{flat_context_mask.get_shape()}')
        # (batch * max_contexts)
        lengths = tf.multiply(tf.reshape(path_lengths, [-1]), tf.cast(flat_context_mask, tf.int32))
        print(f'lengths\n{lengths.get_shape()}')

        # BiRNN:
        rnn_cell_fw = tf.nn.rnn_cell.LSTMCell(self.RNN_SIZE // 2)
        # print(f'rnn_cell_fw\n{rnn_cell_fw.get_shape()}')
        rnn_cell_bw = tf.nn.rnn_cell.LSTMCell(self.RNN_SIZE // 2)
        # print(f'rnn_cell_bw\n{rnn_cell_bw.get_shape()}')
        if is_train:
            rnn_cell_fw = tf.nn.rnn_cell.DropoutWrapper(rnn_cell_fw, output_keep_prob=self.RNN_DROPOUT_KEEP_PROB)
            # print(f'rnn_cell_fw\n{rnn_cell_fw.get_shape()}')
            rnn_cell_bw = tf.nn.rnn_cell.DropoutWrapper(rnn_cell_bw, output_keep_prob=self.RNN_DROPOUT_KEEP_PROB)
            # print(f'rnn_cell_bw\n{rnn_cell_bw.get_shape()}')

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
        # print(f'state_fw\n{state_fw.get_shape()}')
        # print(f'state_bw\n{state_bw.get_shape()}')
        # (batch * max_contexts, rnn_size)
        rnn_final_state = tf.concat([state_fw.h, state_bw.h], axis=-1)
        print(f'rnn_final_state\n{rnn_final_state.get_shape()}')

        # (batch, max_contexts, rnn_size)
        return tf.reshape(rnn_final_state, shape=[-1, max_contexts, self.RNN_SIZE])
