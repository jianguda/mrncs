import pickle

import numpy as np
import tensorflow as tf

PAD = '<PAD>'
UNK = '<UNK>'

_CONTEXT_MASK = '_CONTEXT_MASK'
_LABEL_IDS = '_LABEL_IDS'
_SOURCE_IDS = '_SOURCE_IDS'
_MIDDLE_IDS = '_MIDDLE_IDS'
_TARGET_IDS = '_TARGET_IDS'
_LABEL_LENGTHS = '_LABEL_LENGTHS'
_SOURCE_LENGTHS = '_SOURCE_LENGTHS'
_MIDDLE_LENGTHS = '_MIDDLE_LENGTHS'
_TARGET_LENGTHS = '_TARGET_LENGTHS'


def stats2data(stats):
    token2index, index2token = {}, {}
    size = 0
    for value in [PAD, UNK]:
        token2index[value] = size
        index2token[size] = value
        size += 1
    sorted_counts = [(k, stats[k]) for k in sorted(stats, key=stats.get, reverse=True)]
    limited_sorted = dict(sorted_counts)
    for token, _ in limited_sorted.items():
        token2index[token] = size
        index2token[size] = token
        size += 1
    return token2index, index2token, size


class Model:
    def __init__(self):
        self.config = Config()
        self.sess = tf.Session()

        with open('stats.c2s', 'rb') as file:
            label_stats = pickle.load(file)
            leaf_stats = pickle.load(file)
            node_stats = pickle.load(file)
            print('stats loaded.')

        self.label2index, self.index2label, self.label_size = stats2data(label_stats)
        print('Loaded label token size: %d' % self.label_size)
        self.leaf2index, self.index2leaf, self.leaf_size = stats2data(leaf_stats)
        print('Loaded leaf token size: %d' % self.leaf_size)
        self.node2index, self.index2node, self.node_size = stats2data(node_stats)
        print('Loaded node token size: %d' % self.node_size)

    def train(self):
        print('Starting training')
        reader = Reader(label2index=self.label2index,
                        leaf2index=self.leaf2index,
                        node2index=self.node2index,
                        config=self.config)
        iterator = reader.output_tensors()
        optimizer, loss = self.build_train_graph(iterator.get_next())

        print('Number of trainable params:',
              np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()]))
        # initialize session variables
        self.sess.run(tf.group(
            tf.global_variables_initializer(), tf.local_variables_initializer(), tf.tables_initializer()))

        print('Started reader...')
        for iteration in range(1, self.config.NUM_EPOCHS + 1):
            self.sess.run(iterator.initializer)
            while True:
                _, batch_loss = self.sess.run([optimizer, loss])
                # print('SINGLE BATCH LOSS', batch_loss)

    def build_train_graph(self, input_tensors):
        context_mask = input_tensors[_CONTEXT_MASK]
        label_ids = input_tensors[_LABEL_IDS]
        source_ids = input_tensors[_SOURCE_IDS]
        middle_ids = input_tensors[_MIDDLE_IDS]
        target_ids = input_tensors[_TARGET_IDS]
        label_lengths = input_tensors[_LABEL_LENGTHS]
        source_lengths = input_tensors[_SOURCE_LENGTHS]
        middle_lengths = input_tensors[_MIDDLE_LENGTHS]
        target_lengths = input_tensors[_TARGET_LENGTHS]

        with tf.variable_scope('model'):
            label_token = tf.get_variable(
                'LABEL_TOKEN', shape=(self.label_size, self.config.EMBED_SIZE), dtype=tf.float32,
                initializer=tf.contrib.layers.variance_scaling_initializer(factor=1.0, mode='FAN_OUT', uniform=True))
            leaf_token = tf.get_variable(
                'LEAF_TOKEN', shape=(self.leaf_size, self.config.EMBED_SIZE), dtype=tf.float32,
                initializer=tf.contrib.layers.variance_scaling_initializer(factor=1.0, mode='FAN_OUT', uniform=True))
            node_token = tf.get_variable(
                'NODE_TOKEN', shape=(self.node_size, self.config.EMBED_SIZE), dtype=tf.float32,
                initializer=tf.contrib.layers.variance_scaling_initializer(factor=1.0, mode='FAN_OUT', uniform=True))
            # (batch, max_contexts, vector_size)
            code_vectors, _ = self.compute_contexts(
                leaf_token=leaf_token, node_token=node_token, context_mask=context_mask,
                source_ids=source_ids, middle_ids=middle_ids, target_ids=target_ids,
                source_lengths=source_lengths, middle_lengths=middle_lengths, target_lengths=target_lengths
            )
            logits = tf.matmul(code_vectors, label_token, transpose_b=True)
            crossent = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=label_ids, logits=logits)
            token_words_nonzero = tf.sequence_mask(
                label_lengths + 1, maxlen=self.config.MAX_TOKEN_PARTS + 1, dtype=tf.float32)
            batch_size = tf.shape(label_ids)[0]
            # batch_size = tf.cast(tf.shape(input_tensors.target_index)[0], dtype=tf.float32)
            loss = tf.reduce_sum(crossent * token_words_nonzero) / tf.to_float(batch_size)
            # loss = tf.reduce_sum(tf.nn.sparse_softmax_cross_entropy_with_logits(
            #     labels=tf.reshape(input_tensors.target_index, [-1]),
            #     logits=logits)) / batch_size
            optimizer = tf.train.AdamOptimizer().minimize(loss)
            self.saver = tf.train.Saver(max_to_keep=10)

        return optimizer, loss

    def aggregate_path(self, path_embed, path_lengths, context_mask, is_evaluating=False):
        # path_embed:           (batch, max_contexts, max_path_length+1, dim)
        # path_length:          (batch, max_contexts)
        # context_mask:         (batch, max_contexts)
        max_contexts = tf.shape(path_embed)[1]
        # (batch * max_contexts, max_path_length+1, dim)
        flat_paths = tf.reshape(path_embed, shape=[-1, self.config.MAX_PATH_LENGTH, self.config.EMBED_SIZE])
        # (batch * max_contexts)
        flat_context_mask = tf.reshape(context_mask, [-1])
        # (batch * max_contexts)
        lengths = tf.multiply(tf.reshape(path_lengths, [-1]), tf.cast(flat_context_mask, tf.int32))
        # BiRNN:
        rnn_cell_fw = tf.nn.rnn_cell.LSTMCell(self.config.RNN_SIZE / 2)
        rnn_cell_bw = tf.nn.rnn_cell.LSTMCell(self.config.RNN_SIZE / 2)
        if not is_evaluating:
            rnn_cell_fw = tf.nn.rnn_cell.DropoutWrapper(rnn_cell_fw, output_keep_prob=self.config.RNN_DROPOUT_KEEP_PROB)
            rnn_cell_bw = tf.nn.rnn_cell.DropoutWrapper(rnn_cell_bw, output_keep_prob=self.config.RNN_DROPOUT_KEEP_PROB)
        _, (state_fw, state_bw) = tf.nn.bidirectional_dynamic_rnn(
            cell_fw=rnn_cell_fw, cell_bw=rnn_cell_bw, inputs=flat_paths, dtype=tf.float32, sequence_length=lengths)
        # (batch * max_contexts, rnn_size)
        final_rnn_state = tf.concat([state_fw.h, state_bw.h], axis=-1)

        # (batch, max_contexts, rnn_size)
        return tf.reshape(final_rnn_state, shape=[-1, max_contexts, self.config.RNN_SIZE])

    def compute_contexts(self, leaf_token, node_token, context_mask, source_ids, middle_ids, target_ids,
                         source_lengths, middle_lengths, target_lengths, is_evaluating=False):
        # (batch, max_contexts, max_name_parts, dim)
        source_embed = tf.nn.embedding_lookup(params=leaf_token, ids=source_ids)
        # (batch, max_contexts, max_path_length+1, dim)
        middle_embed = tf.nn.embedding_lookup(params=node_token, ids=middle_ids)
        # (batch, max_contexts, max_name_parts, dim)
        target_embed = tf.nn.embedding_lookup(params=leaf_token, ids=target_ids)

        # (batch, max_contexts, max_name_parts, 1)
        source_mask = tf.expand_dims(
            tf.sequence_mask(source_lengths, maxlen=self.config.MAX_NAME_PARTS, dtype=tf.float32), -1)
        # (batch, max_contexts, max_name_parts, 1)
        target_mask = tf.expand_dims(
            tf.sequence_mask(target_lengths, maxlen=self.config.MAX_NAME_PARTS, dtype=tf.float32), -1)

        # (batch, max_contexts, dim)
        source_token_sum = tf.reduce_sum(source_embed * source_mask, axis=2)
        # (batch, max_contexts, rnn_size)
        middle_aggregation = self.aggregate_path(middle_embed, middle_lengths, context_mask, is_evaluating)
        # (batch, max_contexts, dim)
        target_token_sum = tf.reduce_sum(target_embed * target_mask, axis=2)

        # (batch, max_contexts, dim * 2 + rnn_size)
        context_embed = tf.concat([source_token_sum, middle_aggregation, target_token_sum], axis=-1)
        # (batch, max_contexts, dim * 3)
        # context_embed = tf.concat([source_token_sum, middle_aggregation, target_token_sum], axis=-1)
        if not is_evaluating:
            context_embed = tf.nn.dropout(context_embed, self.config.EMBED_DROPOUT_KEEP_PROB)

        # (batch * max_contexts, dim * 3)
        flat_embed = tf.reshape(context_embed, [-1, self.config.CONTEXT_VECTOR_SIZE])
        transform_shape = (self.config.CONTEXT_VECTOR_SIZE, self.config.CODE_VECTOR_SIZE)
        transform_param = tf.get_variable('TRANSFORM', shape=transform_shape, dtype=tf.float32)

        # (batch * max_contexts, dim * 3)
        flat_embed = tf.tanh(tf.matmul(flat_embed, transform_param))
        attention_shape = (self.config.CODE_VECTOR_SIZE, 1)
        attention_param = tf.get_variable('ATTENTION', shape=attention_shape, dtype=tf.float32)

        # (batch * max_contexts, 1)
        contexts_weights = tf.matmul(flat_embed, attention_param)
        # (batch, max_contexts, 1)
        batched_contexts_weights = tf.reshape(contexts_weights, [-1, self.config.MAX_CONTEXTS, 1])
        # (batch, max_contexts)
        mask = tf.math.log(context_mask)
        # (batch, max_contexts, 1)
        mask = tf.expand_dims(mask, axis=2)
        # (batch, max_contexts, 1)
        batched_contexts_weights += mask
        # (batch, max_contexts, 1)
        attention_weights = tf.nn.softmax(batched_contexts_weights, axis=1)

        batched_shape = (-1, self.config.MAX_CONTEXTS, self.config.CODE_VECTOR_SIZE)
        batched_embed = tf.reshape(flat_embed, shape=batched_shape)
        # (batch, dim * 3)
        code_vectors = tf.reduce_sum(tf.multiply(batched_embed, attention_weights), axis=1)

        return code_vectors, attention_weights

    def predict(self, code_paths):
        print('Starting predicting')
        reader = Reader(label2index=self.label2index,
                        leaf2index=self.leaf2index,
                        node2index=self.node2index,
                        config=self.config,
                        is_evaluating=True)
        reader_output = reader.output_tensors()
        code_vectors, attention_weights = self.build_test_graph(reader_output)

    def build_test_graph(self, input_tensors):
        context_mask = input_tensors[_CONTEXT_MASK]
        label_ids = input_tensors[_LABEL_IDS]
        source_ids = input_tensors[_SOURCE_IDS]
        middle_ids = input_tensors[_MIDDLE_IDS]
        target_ids = input_tensors[_TARGET_IDS]
        label_lengths = input_tensors[_LABEL_LENGTHS]
        source_lengths = input_tensors[_SOURCE_LENGTHS]
        middle_lengths = input_tensors[_MIDDLE_LENGTHS]
        target_lengths = input_tensors[_TARGET_LENGTHS]

        with tf.variable_scope('model', reuse=True):
            label_token = tf.get_variable(
                'LABEL_TOKEN', shape=(self.label_size, self.config.EMBED_SIZE), dtype=tf.float32, trainable=False)
            leaf_token = tf.get_variable(
                'LEAF_TOKEN', shape=(self.leaf_size, self.config.EMBED_SIZE), dtype=tf.float32, trainable=False)
            node_token = tf.get_variable(
                'NODE_TOKEN', shape=(self.node_size, self.config.EMBED_SIZE), dtype=tf.float32, trainable=False)
            # (batch, max_contexts, vector_size)
            code_vectors, attention_weights = self.compute_contexts(
                leaf_token=leaf_token, node_token=node_token, context_mask=context_mask,
                source_ids=source_ids, middle_ids=middle_ids, target_ids=target_ids,
                source_lengths=source_lengths, middle_lengths=middle_lengths, target_lengths=target_lengths,
                is_evaluating=True
            )
        return code_vectors, attention_weights


class Reader:
    def __init__(self, label2index, leaf2index, node2index, config, is_evaluating=False):
        self.config = config
        self.is_evaluating = is_evaluating
        self.label_table = Reader.initialize_table(label2index, label2index[UNK])
        self.leaf_table = Reader.initialize_table(leaf2index, leaf2index[UNK])
        self.node_table = Reader.initialize_table(node2index, node2index[UNK])
        self.context_pad = f'{PAD},{PAD},{PAD}'
        self.file_path = 'test.c2s' if is_evaluating else 'train.c2s'

    @staticmethod
    def initialize_table(token2index, default_value):
        return tf.contrib.lookup.HashTable(
            tf.contrib.lookup.KeyValueTensorInitializer(list(token2index.keys()), list(token2index.values()),
                                                        key_dtype=tf.string, value_dtype=tf.int32), default_value)

    def process_dataset(self, *row_parts):
        row_parts = list(row_parts)
        label_str = row_parts[0]  # (, )

        if not self.is_evaluating:
            all_contexts = tf.stack(row_parts[1:])
            all_contexts_padded = tf.concat([all_contexts, [self.context_pad]], axis=-1)
            index_of_blank_context = tf.where(tf.equal(all_contexts_padded, self.context_pad))
            num_contexts_per_example = tf.reduce_min(index_of_blank_context)
            # if there are less than self.max_contexts valid contexts, still sample self.max_contexts
            safe_limit = tf.cast(tf.maximum(num_contexts_per_example, self.config.MAX_CONTEXTS), tf.int32)
            rand_indices = tf.random_shuffle(tf.range(safe_limit))[:self.config.MAX_CONTEXTS]
            context_str = tf.gather(all_contexts, rand_indices)  # (max_contexts,)
        else:
            context_str = row_parts[1:(self.config.MAX_CONTEXTS + 1)]  # (max_contexts,)

        # labels
        labels = tf.string_split(tf.expand_dims(label_str, -1), delimiter='|')
        shape4labels = [1, tf.maximum(tf.to_int64(self.config.MAX_TOKEN_PARTS), labels.dense_shape[1] + 1)]
        sparse_labels = tf.sparse.SparseTensor(indices=labels.indices,
                                               values=labels.values,
                                               dense_shape=shape4labels)
        dense_labels = tf.reshape(tf.sparse.to_dense(sp_input=sparse_labels, default_value=PAD), [-1])
        ######
        label_ids, label_lengths = self.label2info(dense_labels)
        ######
        # contexts (max_contexts, )
        contexts = tf.string_split(context_str, delimiter=',', skip_empty=False)
        shape4contexts = [self.config.MAX_CONTEXTS, 3]
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

        return {_CONTEXT_MASK: context_mask, _LABEL_IDS: label_ids,
                _SOURCE_IDS: source_ids, _MIDDLE_IDS: middle_ids, _TARGET_IDS: target_ids,
                _LABEL_LENGTHS: label_lengths, _SOURCE_LENGTHS: source_lengths,
                _MIDDLE_LENGTHS: middle_lengths, _TARGET_LENGTHS: target_lengths}

    def label2info(self, labels):
        token_length = tf.reduce_min(tf.where(tf.equal(labels, PAD)))
        dense_token_label = labels[:self.config.MAX_TOKEN_PARTS]
        token_lengths = tf.clip_by_value(token_length, clip_value_min=0,
                                         clip_value_max=self.config.MAX_TOKEN_PARTS)
        # (max_token_parts + 1) of int
        token_ids = tf.concat([self.label_table.lookup(dense_token_label), [0]], axis=-1)
        return token_ids, token_lengths

    def context2info(self, context, index):
        # (max_contexts, 1)
        context_str = tf.slice(context, [0, index], [self.config.MAX_CONTEXTS, 1])
        # (max_contexts)
        flat_context_str = tf.reshape(context_str, [-1])
        # (max_contexts, max_name_parts)
        context_tokens = tf.string_split(flat_context_str, delimiter='|', skip_empty=False)
        if index in (0, 2):
            max_path_length = tf.maximum(tf.to_int64(self.config.MAX_NAME_PARTS), context_tokens.dense_shape[1])
        else:
            max_path_length = self.config.MAX_PATH_LENGTH
        # (batch, max_contexts, max_path_length)
        # (max_contexts, max_path_length)
        sparse_tokens = tf.sparse.SparseTensor(indices=context_tokens.indices, values=context_tokens.values,
                                               dense_shape=[self.config.MAX_CONTEXTS, max_path_length])
        dense_tokens = tf.sparse.to_dense(sp_input=sparse_tokens, default_value=PAD)
        if index in (0, 2):
            dense_tokens = tf.slice(dense_tokens, [0, 0], [-1, self.config.MAX_NAME_PARTS])
            # (max_contexts, max_path_length)
            token_ids = self.leaf_table.lookup(dense_tokens)
        else:
            # (max_contexts, max_path_length)
            token_ids = self.node_table.lookup(dense_tokens)
        # (max_contexts)
        token_lengths = tf.reduce_sum(tf.cast(tf.not_equal(dense_tokens, PAD), tf.int32), -1)
        return token_ids, token_lengths

    def output_tensors(self):
        record_defaults = [[self.context_pad]] * (self.config.MAX_CONTEXTS + 1)
        dataset = tf.data.experimental.CsvDataset(
            self.file_path, record_defaults=record_defaults, field_delim=' ',
            use_quote_delim=False, buffer_size=self.config.CSV_BUFFER_SIZE)
        if not self.is_evaluating:
            dataset = dataset.shuffle(self.config.SHUFFLE_BUFFER_SIZE, reshuffle_each_iteration=True)
        dataset = dataset.apply(tf.data.experimental.map_and_batch(
            map_func=self.process_dataset, batch_size=self.config.BATCH_SIZE))
        dataset = dataset.prefetch(tf.contrib.data.AUTOTUNE)
        iterator = dataset.make_initializable_iterator()
        return iterator


class Config:
    def __init__(self):
        self.NUM_EPOCHS = 1000
        self.BATCH_SIZE = 256
        self.SHUFFLE_BUFFER_SIZE = 10000
        self.CSV_BUFFER_SIZE = 100 * 1024 * 1024  # 100 MB
        self.MAX_CONTEXTS = 200
        self.EMBED_SIZE = 128
        self.RNN_SIZE = 128 * 2  # Two LSTMs to embed paths, each of size 128
        self.MAX_PATH_LENGTH = 8 + 1
        self.MAX_NAME_PARTS = 5
        self.MAX_TOKEN_PARTS = 6
        self.EMBED_DROPOUT_KEEP_PROB = 0.75
        self.RNN_DROPOUT_KEEP_PROB = 0.5
        # The context vector is actually a concatenation of the embedded
        # leaf & token vectors and the embedded path vector.
        self.CONTEXT_VECTOR_SIZE = 3 * self.EMBED_SIZE
        self.CODE_VECTOR_SIZE = 3 * self.EMBED_SIZE


def main():
    np.random.seed(999)
    tf.set_random_seed(999)
    model = Model()
    model.train()
    code_paths = ''
    model.predict(code_paths)


if __name__ == '__main__':
    main()
