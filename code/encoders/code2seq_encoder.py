import _pickle as pickle
import os

import numpy as np
import tensorflow as tf

import reader
from common import Common


class Code2SeqEncoder:
    num_batches_to_log = 100

    def __init__(self, config):
        self.config = config
        self.sess = tf.Session()

        self.subtoken_to_index = None

        if config.LOAD_PATH:
            self.load_model(sess=None)
        else:
            with open('{}.dict.c2s'.format(config.TRAIN_PATH), 'rb') as file:
                subtoken_to_count = pickle.load(file)
                node_to_count = pickle.load(file)
                target_to_count = pickle.load(file)
                max_contexts = pickle.load(file)
                self.num_training_examples = pickle.load(file)
                print('Dictionaries loaded.')

            if self.config.DATA_NUM_CONTEXTS <= 0:
                self.config.DATA_NUM_CONTEXTS = max_contexts
            self.subtoken_to_index, self.index_to_subtoken, self.subtoken_vocab_size = \
                Common.load_vocab_from_dict(subtoken_to_count, add_values=[Common.PAD, Common.UNK],
                                            max_size=config.SUBTOKENS_VOCAB_MAX_SIZE)
            print('Loaded subtoken vocab. size: %d' % self.subtoken_vocab_size)

            self.target_to_index, self.index_to_target, self.target_vocab_size = \
                Common.load_vocab_from_dict(target_to_count, add_values=[Common.PAD, Common.UNK, Common.SOS],
                                            max_size=config.TARGET_VOCAB_MAX_SIZE)
            print('Loaded target word vocab. size: %d' % self.target_vocab_size)

            self.node_to_index, self.index_to_node, self.nodes_vocab_size = \
                Common.load_vocab_from_dict(node_to_count, add_values=[Common.PAD, Common.UNK], max_size=None)
            print('Loaded nodes vocab. size: %d' % self.nodes_vocab_size)
            self.epochs_trained = 0

    def train(self):
        print('Starting training')
        self.queue_thread = reader.Reader(subtoken_to_index=self.subtoken_to_index,
                                          node_to_index=self.node_to_index,
                                          target_to_index=self.target_to_index,
                                          config=self.config)
        optimizer, train_loss = self.build_training_graph(self.queue_thread.get_output())

        print('Number of trainable params:',
              np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()]))
        self.initialize_session_variables(self.sess)

        print('Initalized variables')
        if self.config.LOAD_PATH:
            self.load_model(self.sess)

        print('Started reader...')
        for iteration in range(1, (self.config.NUM_EPOCHS // self.config.SAVE_EVERY_EPOCHS) + 1):
            self.queue_thread.reset(self.sess)
            while True:
                _, batch_loss = self.sess.run([optimizer, train_loss])
                # print('SINGLE BATCH LOSS', batch_loss)

        if self.config.SAVE_PATH:
            self.save_model(self.sess, self.config.SAVE_PATH + '.final')
            print('Model saved in file: %s' % self.config.SAVE_PATH)

    def build_training_graph(self, input_tensors):
        target_index = input_tensors[reader.TARGET_INDEX_KEY]
        target_lengths = input_tensors[reader.TARGET_LENGTH_KEY]
        path_source_indices = input_tensors[reader.PATH_SOURCE_INDICES_KEY]
        node_indices = input_tensors[reader.NODE_INDICES_KEY]
        path_target_indices = input_tensors[reader.PATH_TARGET_INDICES_KEY]
        valid_context_mask = input_tensors[reader.VALID_CONTEXT_MASK_KEY]
        path_source_lengths = input_tensors[reader.PATH_SOURCE_LENGTHS_KEY]
        path_lengths = input_tensors[reader.PATH_LENGTHS_KEY]
        path_target_lengths = input_tensors[reader.PATH_TARGET_LENGTHS_KEY]

        with tf.variable_scope('model'):
            subtoken_vocab = tf.get_variable('SUBTOKENS_VOCAB',
                                             shape=(self.subtoken_vocab_size, self.config.EMBEDDINGS_SIZE),
                                             dtype=tf.float32,
                                             initializer=tf.contrib.layers.variance_scaling_initializer(factor=1.0,
                                                                                                        mode='FAN_OUT',
                                                                                                        uniform=True))
            target_words_vocab = tf.get_variable('TARGET_WORDS_VOCAB',
                                                 shape=(self.target_vocab_size, self.config.EMBEDDINGS_SIZE),
                                                 dtype=tf.float32,
                                                 initializer=tf.contrib.layers.variance_scaling_initializer(factor=1.0,
                                                                                                            mode='FAN_OUT',
                                                                                                            uniform=True))
            nodes_vocab = tf.get_variable('NODES_VOCAB', shape=(self.nodes_vocab_size, self.config.EMBEDDINGS_SIZE),
                                          dtype=tf.float32,
                                          initializer=tf.contrib.layers.variance_scaling_initializer(factor=1.0,
                                                                                                     mode='FAN_OUT',
                                                                                                     uniform=True))
            # (batch, max_contexts, decoder_size)
            batched_contexts = self.compute_contexts(subtoken_vocab=subtoken_vocab, nodes_vocab=nodes_vocab,
                                                     source_input=path_source_indices, nodes_input=node_indices,
                                                     target_input=path_target_indices,
                                                     valid_mask=valid_context_mask,
                                                     path_source_lengths=path_source_lengths,
                                                     path_lengths=path_lengths, path_target_lengths=path_target_lengths)

            batch_size = tf.shape(target_index)[0]
            outputs, final_states = self.attention_output(target_words_vocab=target_words_vocab,
                                                        target_input=target_index, batch_size=batch_size,
                                                        batched_contexts=batched_contexts,
                                                        valid_mask=valid_context_mask)
            step = tf.Variable(0, trainable=False)

            logits = outputs.rnn_output  # (batch, max_output_length, dim * 2 + rnn_size)

            crossent = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=target_index, logits=logits)
            target_words_nonzero = tf.sequence_mask(target_lengths + 1,
                                                    maxlen=self.config.MAX_TARGET_PARTS + 1, dtype=tf.float32)
            loss = tf.reduce_sum(crossent * target_words_nonzero) / tf.to_float(batch_size)

            if self.config.USE_MOMENTUM:
                learning_rate = tf.train.exponential_decay(0.01, step * self.config.BATCH_SIZE,
                                                           self.num_training_examples,
                                                           0.95, staircase=True)
                optimizer = tf.train.MomentumOptimizer(learning_rate, 0.95, use_nesterov=True)
                train_op = optimizer.minimize(loss, global_step=step)
            else:
                params = tf.trainable_variables()
                gradients = tf.gradients(loss, params)
                clipped_gradients, _ = tf.clip_by_global_norm(gradients, clip_norm=5)
                optimizer = tf.train.AdamOptimizer()
                train_op = optimizer.apply_gradients(zip(clipped_gradients, params))

            self.saver = tf.train.Saver(max_to_keep=10)

        return train_op, loss

    def attention_output(self, target_words_vocab, target_input, batch_size, batched_contexts, valid_mask):
        num_contexts_per_example = tf.count_nonzero(valid_mask, axis=-1)

        start_fill = tf.fill([batch_size],
                             self.target_to_index[Common.SOS])  # (batch, )
        decoder_cell = tf.nn.rnn_cell.MultiRNNCell([
            tf.nn.rnn_cell.LSTMCell(self.config.DECODER_SIZE) for _ in range(self.config.NUM_DECODER_LAYERS)
        ])
        contexts_sum = tf.reduce_sum(batched_contexts * tf.expand_dims(valid_mask, -1),
                                     axis=1)  # (batch_size, dim * 2 + rnn_size)
        contexts_average = tf.divide(contexts_sum, tf.to_float(tf.expand_dims(num_contexts_per_example, -1)))
        fake_encoder_state = tuple(tf.nn.rnn_cell.LSTMStateTuple(contexts_average, contexts_average) for _ in
                                   range(self.config.NUM_DECODER_LAYERS))
        projection_layer = tf.layers.Dense(self.target_vocab_size, use_bias=False)

        attention_mechanism = tf.contrib.seq2seq.LuongAttention(
            num_units=self.config.DECODER_SIZE,
            memory=batched_contexts
        )
        # TF doesn't support beam search with alignment history
        decoder_cell = tf.contrib.seq2seq.AttentionWrapper(decoder_cell, attention_mechanism,
                                                           attention_layer_size=self.config.DECODER_SIZE,
                                                           alignment_history=False)

        decoder_cell = tf.nn.rnn_cell.DropoutWrapper(decoder_cell,
                                                     output_keep_prob=self.config.RNN_DROPOUT_KEEP_PROB)
        target_words_embedding = tf.nn.embedding_lookup(target_words_vocab,
                                                        tf.concat([tf.expand_dims(start_fill, -1), target_input],
                                                                  axis=-1))  # (batch, max_target_parts, dim * 2 + rnn_size)
        helper = tf.contrib.seq2seq.TrainingHelper(inputs=target_words_embedding,
                                                   sequence_length=tf.ones([batch_size], dtype=tf.int32) * (
                                                           self.config.MAX_TARGET_PARTS + 1))
        initial_state = decoder_cell.zero_state(batch_size, tf.float32).clone(cell_state=fake_encoder_state)
        decoder = tf.contrib.seq2seq.BasicDecoder(cell=decoder_cell, helper=helper, initial_state=initial_state,
                                                  output_layer=projection_layer)

        outputs, final_states, final_sequence_lengths = tf.contrib.seq2seq.dynamic_decode(decoder,
                                                                                          maximum_iterations=self.config.MAX_TARGET_PARTS + 1)
        return outputs, final_states

    def calculate_path_abstraction(self, path_embed, path_lengths, valid_contexts_mask):
        # path_embed:           (batch, max_contexts, max_path_length+1, dim)
        # path_length:          (batch, max_contexts)
        # valid_contexts_mask:  (batch, max_contexts)
        max_contexts = tf.shape(path_embed)[1]
        flat_paths = tf.reshape(path_embed, shape=[-1, self.config.MAX_PATH_LENGTH,
                                                   self.config.EMBEDDINGS_SIZE])  # (batch * max_contexts, max_path_length+1, dim)
        flat_valid_contexts_mask = tf.reshape(valid_contexts_mask, [-1])  # (batch * max_contexts)
        lengths = tf.multiply(tf.reshape(path_lengths, [-1]),
                              tf.cast(flat_valid_contexts_mask, tf.int32))  # (batch * max_contexts)
        if self.config.BIRNN:
            rnn_cell_fw = tf.nn.rnn_cell.LSTMCell(self.config.RNN_SIZE / 2)
            rnn_cell_bw = tf.nn.rnn_cell.LSTMCell(self.config.RNN_SIZE / 2)

            rnn_cell_fw = tf.nn.rnn_cell.DropoutWrapper(rnn_cell_fw,
                                                        output_keep_prob=self.config.RNN_DROPOUT_KEEP_PROB)
            rnn_cell_bw = tf.nn.rnn_cell.DropoutWrapper(rnn_cell_bw,
                                                        output_keep_prob=self.config.RNN_DROPOUT_KEEP_PROB)
            _, (state_fw, state_bw) = tf.nn.bidirectional_dynamic_rnn(
                cell_fw=rnn_cell_fw,
                cell_bw=rnn_cell_bw,
                inputs=flat_paths,
                dtype=tf.float32,
                sequence_length=lengths)
            final_rnn_state = tf.concat([state_fw.h, state_bw.h], axis=-1)  # (batch * max_contexts, rnn_size)
        else:
            rnn_cell = tf.nn.rnn_cell.LSTMCell(self.config.RNN_SIZE)
            rnn_cell = tf.nn.rnn_cell.DropoutWrapper(rnn_cell, output_keep_prob=self.config.RNN_DROPOUT_KEEP_PROB)
            _, state = tf.nn.dynamic_rnn(
                cell=rnn_cell,
                inputs=flat_paths,
                dtype=tf.float32,
                sequence_length=lengths
            )
            final_rnn_state = state.h  # (batch * max_contexts, rnn_size)

        return tf.reshape(final_rnn_state,
                          shape=[-1, max_contexts, self.config.RNN_SIZE])  # (batch, max_contexts, rnn_size)

    def compute_contexts(self, subtoken_vocab, nodes_vocab, source_input, nodes_input,
                         target_input, valid_mask, path_source_lengths, path_lengths, path_target_lengths):

        source_word_embed = tf.nn.embedding_lookup(params=subtoken_vocab,
                                                   ids=source_input)  # (batch, max_contexts, max_name_parts, dim)
        path_embed = tf.nn.embedding_lookup(params=nodes_vocab,
                                            ids=nodes_input)  # (batch, max_contexts, max_path_length+1, dim)
        target_word_embed = tf.nn.embedding_lookup(params=subtoken_vocab,
                                                   ids=target_input)  # (batch, max_contexts, max_name_parts, dim)

        source_word_mask = tf.expand_dims(
            tf.sequence_mask(path_source_lengths, maxlen=self.config.MAX_NAME_PARTS, dtype=tf.float32),
            -1)  # (batch, max_contexts, max_name_parts, 1)
        target_word_mask = tf.expand_dims(
            tf.sequence_mask(path_target_lengths, maxlen=self.config.MAX_NAME_PARTS, dtype=tf.float32),
            -1)  # (batch, max_contexts, max_name_parts, 1)

        source_words_sum = tf.reduce_sum(source_word_embed * source_word_mask, axis=2)  # (batch, max_contexts, dim)
        path_nodes_aggregation = self.calculate_path_abstraction(path_embed, path_lengths, valid_mask)  # (batch, max_contexts, rnn_size)
        target_words_sum = tf.reduce_sum(target_word_embed * target_word_mask, axis=2)  # (batch, max_contexts, dim)

        context_embed = tf.concat([source_words_sum, path_nodes_aggregation, target_words_sum], axis=-1)  # (batch, max_contexts, dim * 2 + rnn_size)
        context_embed = tf.nn.dropout(context_embed, self.config.EMBEDDINGS_DROPOUT_KEEP_PROB)

        batched_embed = tf.layers.dense(inputs=context_embed, units=self.config.DECODER_SIZE,
                                        activation=tf.nn.tanh, trainable=True, use_bias=False)

        return batched_embed

    def save_model(self, sess, path):
        save_target = path + '_iter%d' % self.epochs_trained
        dirname = os.path.dirname(save_target)
        if not os.path.exists(dirname):
            os.makedirs(dirname)
        self.saver.save(sess, save_target)

        dictionaries_path = save_target + '.dict'
        with open(dictionaries_path, 'wb') as file:
            pickle.dump(self.subtoken_to_index, file)
            pickle.dump(self.index_to_subtoken, file)
            pickle.dump(self.subtoken_vocab_size, file)

            pickle.dump(self.target_to_index, file)
            pickle.dump(self.index_to_target, file)
            pickle.dump(self.target_vocab_size, file)

            pickle.dump(self.node_to_index, file)
            pickle.dump(self.index_to_node, file)
            pickle.dump(self.nodes_vocab_size, file)

            pickle.dump(self.num_training_examples, file)
            pickle.dump(self.epochs_trained, file)
            pickle.dump(self.config, file)
        print('Saved after %d epochs in: %s' % (self.epochs_trained, save_target))

    def load_model(self, sess):
        if not sess is None:
            self.saver.restore(sess, self.config.LOAD_PATH)
            print('Done loading model')
        with open(self.config.LOAD_PATH + '.dict', 'rb') as file:
            if self.subtoken_to_index is not None:
                return
            print('Loading dictionaries from: ' + self.config.LOAD_PATH)
            self.subtoken_to_index = pickle.load(file)
            self.index_to_subtoken = pickle.load(file)
            self.subtoken_vocab_size = pickle.load(file)

            self.target_to_index = pickle.load(file)
            self.index_to_target = pickle.load(file)
            self.target_vocab_size = pickle.load(file)

            self.node_to_index = pickle.load(file)
            self.index_to_node = pickle.load(file)
            self.nodes_vocab_size = pickle.load(file)

            self.num_training_examples = pickle.load(file)
            self.epochs_trained = pickle.load(file)
            saved_config = pickle.load(file)
            self.config.take_model_hyperparams_from(saved_config)
            print('Done loading dictionaries')

    @staticmethod
    def initialize_session_variables(sess):
        sess.run(tf.group(tf.global_variables_initializer(), tf.local_variables_initializer(), tf.tables_initializer()))
