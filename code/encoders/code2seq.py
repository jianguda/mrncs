import _pickle as pickle
import os

import numpy as np
import tensorflow as tf

SOS = '<S>'
PAD = '<PAD>'
UNK = '<UNK>'

TARGET_INDEX_KEY = 'TARGET_INDEX_KEY'
TARGET_STRING_KEY = 'TARGET_STRING_KEY'
TARGET_LENGTH_KEY = 'TARGET_LENGTH_KEY'
PATH_SOURCE_INDICES_KEY = 'PATH_SOURCE_INDICES_KEY'
NODE_INDICES_KEY = 'NODES_INDICES_KEY'
PATH_TARGET_INDICES_KEY = 'PATH_TARGET_INDICES_KEY'
VALID_CONTEXT_MASK_KEY = 'VALID_CONTEXT_MASK_KEY'
PATH_SOURCE_LENGTHS_KEY = 'PATH_SOURCE_LENGTHS_KEY'
PATH_LENGTHS_KEY = 'PATH_LENGTHS_KEY'
PATH_TARGET_LENGTHS_KEY = 'PATH_TARGET_LENGTHS_KEY'
PATH_SOURCE_STRINGS_KEY = 'PATH_SOURCE_STRINGS_KEY'
PATH_STRINGS_KEY = 'PATH_STRINGS_KEY'
PATH_TARGET_STRINGS_KEY = 'PATH_TARGET_STRINGS_KEY'


def load_vocab_from_dict(word_to_count, add_values=[], max_size=None):
    word_to_index, index_to_word = {}, {}
    current_index = 0
    for value in add_values:
        word_to_index[value] = current_index
        index_to_word[current_index] = value
        current_index += 1
    sorted_counts = [(k, word_to_count[k]) for k in sorted(word_to_count, key=word_to_count.get, reverse=True)]
    limited_sorted = dict(sorted_counts[:max_size])
    for word, count in limited_sorted.items():
        word_to_index[word] = current_index
        index_to_word[current_index] = word
        current_index += 1
    return word_to_index, index_to_word, current_index


class Model:
    def __init__(self):
        self.config = Config()
        self.sess = tf.Session()
        self.subtoken_to_index = None

        with open('{}.dict.c2s'.format(self.config.DATA_PATH), 'rb') as file:
            subtoken_to_count = pickle.load(file)
            node_to_count = pickle.load(file)
            target_to_count = pickle.load(file)
            max_contexts = pickle.load(file)
            print('Dictionaries loaded.')

        if self.config.DATA_NUM_CONTEXTS <= 0:
            self.config.DATA_NUM_CONTEXTS = max_contexts
        self.subtoken_to_index, self.index_to_subtoken, self.subtoken_vocab_size = \
            load_vocab_from_dict(subtoken_to_count, add_values=[PAD, UNK],
                                 max_size=self.config.SUBTOKENS_VOCAB_MAX_SIZE)
        print('Loaded subtoken vocab. size: %d' % self.subtoken_vocab_size)

        self.target_to_index, self.index_to_target, self.target_vocab_size = \
            load_vocab_from_dict(target_to_count, add_values=[PAD, UNK, SOS],
                                 max_size=self.config.TARGET_VOCAB_MAX_SIZE)
        print('Loaded target word vocab. size: %d' % self.target_vocab_size)

        self.node_to_index, self.index_to_node, self.nodes_vocab_size = \
            load_vocab_from_dict(node_to_count, add_values=[PAD, UNK], max_size=None)
        print('Loaded nodes vocab. size: %d' % self.nodes_vocab_size)
        self.epochs_trained = 0

    def train(self):
        print('Starting training')
        self.queue_thread = Reader(subtoken_to_index=self.subtoken_to_index,
                                   node_to_index=self.node_to_index,
                                   target_to_index=self.target_to_index,
                                   config=self.config)
        optimizer, train_loss = self.build_training_graph(self.queue_thread.get_output())

        print('Number of trainable params:',
              np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()]))
        self.initialize_session_variables(self.sess)

        print('Started reader...')
        for iteration in range(1, (self.config.NUM_EPOCHS // self.config.SAVE_EVERY_EPOCHS) + 1):
            self.queue_thread.reset(self.sess)
            while True:
                _, batch_loss = self.sess.run([optimizer, train_loss])
                # print('SINGLE BATCH LOSS', batch_loss)

    def build_training_graph(self, input_tensors):
        target_index = input_tensors[TARGET_INDEX_KEY]
        target_lengths = input_tensors[TARGET_LENGTH_KEY]
        path_source_indices = input_tensors[PATH_SOURCE_INDICES_KEY]
        node_indices = input_tensors[NODE_INDICES_KEY]
        path_target_indices = input_tensors[PATH_TARGET_INDICES_KEY]
        valid_context_mask = input_tensors[VALID_CONTEXT_MASK_KEY]
        path_source_lengths = input_tensors[PATH_SOURCE_LENGTHS_KEY]
        path_lengths = input_tensors[PATH_LENGTHS_KEY]
        path_target_lengths = input_tensors[PATH_TARGET_LENGTHS_KEY]

        with tf.variable_scope('model'):
            subtoken_vocab = tf.get_variable(
                'SUBTOKENS_VOCAB', shape=(self.subtoken_vocab_size, self.config.EMBEDDINGS_SIZE), dtype=tf.float32,
                initializer=tf.contrib.layers.variance_scaling_initializer(factor=1.0, mode='FAN_OUT', uniform=True))
            target_words_vocab = tf.get_variable(
                'TARGET_WORDS_VOCAB', shape=(self.target_vocab_size, self.config.EMBEDDINGS_SIZE), dtype=tf.float32,
                initializer=tf.contrib.layers.variance_scaling_initializer(factor=1.0, mode='FAN_OUT', uniform=True))
            nodes_vocab = tf.get_variable(
                'NODES_VOCAB', shape=(self.nodes_vocab_size, self.config.EMBEDDINGS_SIZE), dtype=tf.float32,
                initializer=tf.contrib.layers.variance_scaling_initializer(factor=1.0, mode='FAN_OUT', uniform=True))
            # (batch, max_contexts, decoder_size)
            code_vectors = self.compute_contexts(
                subtoken_vocab=subtoken_vocab, nodes_vocab=nodes_vocab,
                source_input=path_source_indices, nodes_input=node_indices,
                target_input=path_target_indices, valid_mask=valid_context_mask,
                path_source_lengths=path_source_lengths, path_lengths=path_lengths,
                path_target_lengths=path_target_lengths)

            batch_size = tf.shape(target_index)[0]

            logits = tf.matmul(code_vectors, target_words_vocab, transpose_b=True)
            crossent = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=target_index, logits=logits)
            target_words_nonzero = tf.sequence_mask(
                target_lengths + 1, maxlen=self.config.MAX_TARGET_PARTS + 1, dtype=tf.float32)
            loss = tf.reduce_sum(crossent * target_words_nonzero) / tf.to_float(batch_size)

            optimizer = tf.train.AdamOptimizer().minimize(loss)

            self.saver = tf.train.Saver(max_to_keep=10)

        return optimizer, loss

    def calculate_path_abstraction(self, path_embed, path_lengths, valid_contexts_mask):
        # path_embed:           (batch, max_contexts, max_path_length+1, dim)
        # path_length:          (batch, max_contexts)
        # valid_contexts_mask:  (batch, max_contexts)
        max_contexts = tf.shape(path_embed)[1]
        # (batch * max_contexts, max_path_length+1, dim)
        flat_paths = tf.reshape(path_embed, shape=[-1, self.config.MAX_PATH_LENGTH, self.config.EMBEDDINGS_SIZE])
        # (batch * max_contexts)
        flat_valid_contexts_mask = tf.reshape(valid_contexts_mask, [-1])
        # (batch * max_contexts)
        lengths = tf.multiply(tf.reshape(path_lengths, [-1]), tf.cast(flat_valid_contexts_mask, tf.int32))
        # BiRNN:
        rnn_cell_fw = tf.nn.rnn_cell.LSTMCell(self.config.RNN_SIZE / 2)
        rnn_cell_bw = tf.nn.rnn_cell.LSTMCell(self.config.RNN_SIZE / 2)
        rnn_cell_fw = tf.nn.rnn_cell.DropoutWrapper(rnn_cell_fw, output_keep_prob=self.config.RNN_DROPOUT_KEEP_PROB)
        rnn_cell_bw = tf.nn.rnn_cell.DropoutWrapper(rnn_cell_bw, output_keep_prob=self.config.RNN_DROPOUT_KEEP_PROB)
        _, (state_fw, state_bw) = tf.nn.bidirectional_dynamic_rnn(
            cell_fw=rnn_cell_fw, cell_bw=rnn_cell_bw, inputs=flat_paths, dtype=tf.float32, sequence_length=lengths)
        # (batch * max_contexts, rnn_size)
        final_rnn_state = tf.concat([state_fw.h, state_bw.h], axis=-1)

        # (batch, max_contexts, rnn_size)
        return tf.reshape(final_rnn_state, shape=[-1, max_contexts, self.config.RNN_SIZE])

    def compute_contexts(self, subtoken_vocab, nodes_vocab, source_input, nodes_input,
                         target_input, valid_mask, path_source_lengths, path_lengths, path_target_lengths):
        # (batch, max_contexts, max_name_parts, dim)
        source_word_embed = tf.nn.embedding_lookup(params=subtoken_vocab, ids=source_input)
        # (batch, max_contexts, max_path_length+1, dim)
        path_embed = tf.nn.embedding_lookup(params=nodes_vocab, ids=nodes_input)
        # (batch, max_contexts, max_name_parts, dim)
        target_word_embed = tf.nn.embedding_lookup(params=subtoken_vocab, ids=target_input)

        # (batch, max_contexts, max_name_parts, 1)
        source_word_mask = tf.expand_dims(
            tf.sequence_mask(path_source_lengths, maxlen=self.config.MAX_NAME_PARTS, dtype=tf.float32), -1)
        # (batch, max_contexts, max_name_parts, 1)
        target_word_mask = tf.expand_dims(
            tf.sequence_mask(path_target_lengths, maxlen=self.config.MAX_NAME_PARTS, dtype=tf.float32), -1)

        # (batch, max_contexts, dim)
        source_words_sum = tf.reduce_sum(source_word_embed * source_word_mask, axis=2)
        # (batch, max_contexts, rnn_size)
        path_nodes_aggregation = self.calculate_path_abstraction(path_embed, path_lengths, valid_mask)
        # (batch, max_contexts, dim)
        target_words_sum = tf.reduce_sum(target_word_embed * target_word_mask, axis=2)

        # (batch, max_contexts, dim * 2 + rnn_size)
        context_embed = tf.concat([source_words_sum, path_nodes_aggregation, target_words_sum], axis=-1)
        context_embed = tf.nn.dropout(context_embed, self.config.EMBEDDINGS_DROPOUT_KEEP_PROB)

        # (batch * max_contexts, dim * 3)
        flat_embed = tf.reshape(context_embed, [-1, self.config.context_vector_size])
        transform_param = tf.get_variable('TRANSFORM',
                                          shape=(self.config.context_vector_size, self.config.CODE_VECTOR_SIZE),
                                          dtype=tf.float32)

        # (batch * max_contexts, dim * 3)
        flat_embed = tf.tanh(tf.matmul(flat_embed, transform_param))

        attention_param = tf.get_variable('ATTENTION', shape=(self.config.CODE_VECTOR_SIZE, 1), dtype=tf.float32)
        # (batch * max_contexts, 1)
        contexts_weights = tf.matmul(flat_embed, attention_param)
        # (batch, max_contexts, 1)
        batched_contexts_weights = tf.reshape(contexts_weights, [-1, self.config.MAX_CONTEXTS, 1])
        # (batch, max_contexts)
        mask = tf.math.log(valid_mask)
        # (batch, max_contexts, 1)
        mask = tf.expand_dims(mask, axis=2)
        # (batch, max_contexts, 1)
        batched_contexts_weights += mask
        # (batch, max_contexts, 1)
        attention_weights = tf.nn.softmax(batched_contexts_weights, axis=1)

        batched_embed = tf.reshape(flat_embed, shape=[-1, self.config.MAX_CONTEXTS, self.config.CODE_VECTOR_SIZE])
        # (batch, dim * 3)
        code_vectors = tf.reduce_sum(tf.multiply(batched_embed, attention_weights), axis=1)

        # , attention_weights
        return code_vectors

    @staticmethod
    def initialize_session_variables(sess):
        sess.run(tf.group(tf.global_variables_initializer(), tf.local_variables_initializer(), tf.tables_initializer()))


class Reader:
    class_subtoken_table = None
    class_target_table = None
    class_node_table = None

    def __init__(self, subtoken_to_index, target_to_index, node_to_index, config):
        self.config = config
        self.file_path = config.DATA_PATH + '.train.c2s'
        if self.file_path is not None and not os.path.exists(self.file_path):
            print(
                '%s cannot find file: %s' % ('Train reader', self.file_path))
        self.batch_size = config.BATCH_SIZE

        self.context_pad = '{},{},{}'.format(PAD, PAD, PAD)
        self.record_defaults = [[self.context_pad]] * (self.config.DATA_NUM_CONTEXTS + 1)

        self.subtoken_table = Reader.get_subtoken_table(subtoken_to_index)
        self.target_table = Reader.get_target_table(target_to_index)
        self.node_table = Reader.get_node_table(node_to_index)
        if self.file_path is not None:
            self.output_tensors = self.compute_output()

    @classmethod
    def get_subtoken_table(cls, subtoken_to_index):
        if cls.class_subtoken_table is None:
            cls.class_subtoken_table = cls.initialize_hash_map(subtoken_to_index, subtoken_to_index[UNK])
        return cls.class_subtoken_table

    @classmethod
    def get_target_table(cls, target_to_index):
        if cls.class_target_table is None:
            cls.class_target_table = cls.initialize_hash_map(target_to_index, target_to_index[UNK])
        return cls.class_target_table

    @classmethod
    def get_node_table(cls, node_to_index):
        if cls.class_node_table is None:
            cls.class_node_table = cls.initialize_hash_map(node_to_index, node_to_index[UNK])
        return cls.class_node_table

    @classmethod
    def initialize_hash_map(cls, word_to_index, default_value):
        return tf.contrib.lookup.HashTable(
            tf.contrib.lookup.KeyValueTensorInitializer(list(word_to_index.keys()), list(word_to_index.values()),
                                                        key_dtype=tf.string,
                                                        value_dtype=tf.int32), default_value)

    def process_dataset(self, *row_parts):
        row_parts = list(row_parts)
        word = row_parts[0]  # (, )

        contexts = row_parts[1:(self.config.MAX_CONTEXTS + 1)]  # (max_contexts,)
        # contexts: (max_contexts, )
        split_contexts = tf.string_split(contexts, delimiter=',', skip_empty=False)
        sparse_split_contexts = tf.sparse.SparseTensor(indices=split_contexts.indices,
                                                       values=split_contexts.values,
                                                       dense_shape=[self.config.MAX_CONTEXTS, 3])
        dense_split_contexts = tf.reshape(
            tf.sparse.to_dense(sp_input=sparse_split_contexts, default_value=PAD),
            shape=[self.config.MAX_CONTEXTS, 3])  # (batch, max_contexts, 3)

        split_target_labels = tf.string_split(tf.expand_dims(word, -1), delimiter='|')
        target_dense_shape = [1, tf.maximum(tf.to_int64(self.config.MAX_TARGET_PARTS),
                                            split_target_labels.dense_shape[1] + 1)]
        sparse_target_labels = tf.sparse.SparseTensor(indices=split_target_labels.indices,
                                                      values=split_target_labels.values,
                                                      dense_shape=target_dense_shape)
        dense_target_label = tf.reshape(tf.sparse.to_dense(sp_input=sparse_target_labels,
                                                           default_value=PAD), [-1])
        index_of_blank = tf.where(tf.equal(dense_target_label, PAD))
        target_length = tf.reduce_min(index_of_blank)
        dense_target_label = dense_target_label[:self.config.MAX_TARGET_PARTS]
        clipped_target_lengths = tf.clip_by_value(target_length, clip_value_min=0,
                                                  clip_value_max=self.config.MAX_TARGET_PARTS)
        target_word_labels = tf.concat([
            self.target_table.lookup(dense_target_label), [0]], axis=-1)  # (max_target_parts + 1) of int

        path_source_strings = tf.slice(dense_split_contexts, [0, 0], [self.config.MAX_CONTEXTS, 1])  # (max_contexts, 1)
        flat_source_strings = tf.reshape(path_source_strings, [-1])  # (max_contexts)
        split_source = tf.string_split(flat_source_strings, delimiter='|',
                                       skip_empty=False)  # (max_contexts, max_name_parts)

        sparse_split_source = tf.sparse.SparseTensor(indices=split_source.indices, values=split_source.values,
                                                     dense_shape=[self.config.MAX_CONTEXTS,
                                                                  tf.maximum(tf.to_int64(self.config.MAX_NAME_PARTS),
                                                                             split_source.dense_shape[1])])
        dense_split_source = tf.sparse.to_dense(sp_input=sparse_split_source,
                                                default_value=PAD)  # (max_contexts, max_name_parts)
        dense_split_source = tf.slice(dense_split_source, [0, 0], [-1, self.config.MAX_NAME_PARTS])
        path_source_indices = self.subtoken_table.lookup(dense_split_source)  # (max_contexts, max_name_parts)
        path_source_lengths = tf.reduce_sum(tf.cast(tf.not_equal(dense_split_source, PAD), tf.int32),
                                            -1)  # (max_contexts)

        path_strings = tf.slice(dense_split_contexts, [0, 1], [self.config.MAX_CONTEXTS, 1])
        flat_path_strings = tf.reshape(path_strings, [-1])
        split_path = tf.string_split(flat_path_strings, delimiter='|', skip_empty=False)
        sparse_split_path = tf.sparse.SparseTensor(indices=split_path.indices, values=split_path.values,
                                                   dense_shape=[self.config.MAX_CONTEXTS, self.config.MAX_PATH_LENGTH])
        dense_split_path = tf.sparse.to_dense(sp_input=sparse_split_path,
                                              default_value=PAD)  # (batch, max_contexts, max_path_length)

        node_indices = self.node_table.lookup(dense_split_path)  # (max_contexts, max_path_length)
        path_lengths = tf.reduce_sum(tf.cast(tf.not_equal(dense_split_path, PAD), tf.int32),
                                     -1)  # (max_contexts)

        path_target_strings = tf.slice(dense_split_contexts, [0, 2], [self.config.MAX_CONTEXTS, 1])  # (max_contexts, 1)
        flat_target_strings = tf.reshape(path_target_strings, [-1])  # (max_contexts)
        split_target = tf.string_split(flat_target_strings, delimiter='|',
                                       skip_empty=False)  # (max_contexts, max_name_parts)
        sparse_split_target = tf.sparse.SparseTensor(indices=split_target.indices, values=split_target.values,
                                                     dense_shape=[self.config.MAX_CONTEXTS,
                                                                  tf.maximum(tf.to_int64(self.config.MAX_NAME_PARTS),
                                                                             split_target.dense_shape[1])])
        dense_split_target = tf.sparse.to_dense(sp_input=sparse_split_target,
                                                default_value=PAD)  # (max_contexts, max_name_parts)
        dense_split_target = tf.slice(dense_split_target, [0, 0], [-1, self.config.MAX_NAME_PARTS])
        path_target_indices = self.subtoken_table.lookup(dense_split_target)  # (max_contexts, max_name_parts)
        path_target_lengths = tf.reduce_sum(tf.cast(tf.not_equal(dense_split_target, PAD), tf.int32),
                                            -1)  # (max_contexts)

        valid_contexts_mask = tf.to_float(tf.not_equal(
            tf.reduce_max(path_source_indices, -1) + tf.reduce_max(node_indices, -1) + tf.reduce_max(
                path_target_indices, -1), 0))

        return {TARGET_STRING_KEY: word, TARGET_INDEX_KEY: target_word_labels,
                TARGET_LENGTH_KEY: clipped_target_lengths,
                PATH_SOURCE_INDICES_KEY: path_source_indices, NODE_INDICES_KEY: node_indices,
                PATH_TARGET_INDICES_KEY: path_target_indices, VALID_CONTEXT_MASK_KEY: valid_contexts_mask,
                PATH_SOURCE_LENGTHS_KEY: path_source_lengths, PATH_LENGTHS_KEY: path_lengths,
                PATH_TARGET_LENGTHS_KEY: path_target_lengths, PATH_SOURCE_STRINGS_KEY: path_source_strings,
                PATH_STRINGS_KEY: path_strings, PATH_TARGET_STRINGS_KEY: path_target_strings
                }

    def reset(self, sess):
        sess.run(self.reset_op)

    def get_output(self):
        return self.output_tensors

    def compute_output(self):
        dataset = tf.data.experimental.CsvDataset(self.file_path, record_defaults=self.record_defaults, field_delim=' ',
                                                  use_quote_delim=False, buffer_size=self.config.CSV_BUFFER_SIZE)

        if self.config.SAVE_EVERY_EPOCHS > 1:
            dataset = dataset.repeat(self.config.SAVE_EVERY_EPOCHS)
        dataset = dataset.shuffle(self.config.SHUFFLE_BUFFER_SIZE, reshuffle_each_iteration=True)
        dataset = dataset.apply(tf.data.experimental.map_and_batch(
            map_func=self.process_dataset, batch_size=self.batch_size,
            num_parallel_batches=self.config.READER_NUM_PARALLEL_BATCHES))
        dataset = dataset.prefetch(tf.contrib.data.AUTOTUNE)
        self.iterator = dataset.make_initializable_iterator()
        self.reset_op = self.iterator.initializer
        return self.iterator.get_next()


class Config:
    def __init__(self):
        self.DATA_PATH = ''
        self.DATA_NUM_CONTEXTS = 0
        self.NUM_EPOCHS = 1000
        self.SAVE_EVERY_EPOCHS = 1
        self.BATCH_SIZE = 256
        self.READER_NUM_PARALLEL_BATCHES = 1
        self.SHUFFLE_BUFFER_SIZE = 10000
        self.CSV_BUFFER_SIZE = 100 * 1024 * 1024  # 100 MB
        self.MAX_CONTEXTS = 200
        self.SUBTOKENS_VOCAB_MAX_SIZE = 190000
        self.TARGET_VOCAB_MAX_SIZE = 27000
        self.EMBEDDINGS_SIZE = 128
        self.RNN_SIZE = 128 * 2  # Two LSTMs to embed paths, each of size 128
        self.MAX_PATH_LENGTH = 8 + 1
        self.MAX_NAME_PARTS = 5
        self.MAX_TARGET_PARTS = 6
        self.EMBEDDINGS_DROPOUT_KEEP_PROB = 0.75
        self.RNN_DROPOUT_KEEP_PROB = 0.5
        # The context vector is actually a concatenation of the embedded
        # source & target vectors and the embedded path vector.
        self.CODE_VECTOR_SIZE = 3 * self.EMBEDDINGS_SIZE


if __name__ == '__main__':
    np.random.seed(239)
    tf.set_random_seed(239)
    Model().train()
