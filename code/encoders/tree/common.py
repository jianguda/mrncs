from typing import Dict, Any, Union, Tuple

import tensorflow as tf
from utils.tfutils import get_activation, write_to_feed_dict, pool_sequence_embedding


class Common:
    @staticmethod
    def yet_attention_layer(embedding):
        hidden_dim = embedding.shape[2].value
        return Common.self_attention_layer(embedding, hidden_dim)
        # # (B,T,D)
        # # [batch_size, seq_length, hidden_size]
        # hidden_size = embedding.shape[2].value  # D value - hidden size
        # attention_size = 128
        # initializer = tf.random_normal_initializer(stddev=0.1)
        #
        # # Trainable parameters
        # w_omega = tf.get_variable(name="w_omega", shape=[hidden_size, attention_size], initializer=initializer)
        # b_omega = tf.get_variable(name="b_omega", shape=[attention_size], initializer=initializer)
        # u_omega = tf.get_variable(name="u_omega", shape=[attention_size], initializer=initializer)
        #
        # with tf.name_scope('att'):
        #     # Applying fully connected layer with non-linear activation to each of the B*T timestamps;
        #     #  the shape of `v` is (B,T,D)*(D,A)=(B,T,A), where A=attention_size
        #     v = tf.tanh(tf.tensordot(embedding, w_omega, axes=1) + b_omega)
        #
        # # For each of the timestamps its vector of size A from `v` is reduced with `u` vector
        # vu = tf.tensordot(v, u_omega, axes=1, name='vu')  # (B,T) shape
        # alphas = tf.nn.softmax(vu, name='alphas')  # (B,T) shape
        #
        # # Output of (Bi-)RNN is reduced with attention vector; the result has (B,D) shape
        # output = tf.reduce_sum(embedding * tf.expand_dims(alphas, -1), 1)
        # return output

    @staticmethod
    def self_attention_layer(inputs, hidden_dim=64, seq_len=128):
        batch_size = 1
        input_seq_len = seq_len
        output_seq_len = seq_len
        # inputs = tf.random_normal((batch_size, input_seq_len, hidden_dim))
        Q_layer = tf.layers.dense(inputs, hidden_dim)  # [batch_size, input_seq_len, hidden_dim]
        K_layer = tf.layers.dense(inputs, hidden_dim)  # [batch_size, input_seq_len, hidden_dim]
        V_layer = tf.layers.dense(inputs, output_seq_len)  # [batch_size, input_seq_len, output_seq_len]
        # attention function
        # [batch_size, input_seq_len, input_seq_len]
        attention = tf.matmul(Q_layer, K_layer, transpose_b=True)
        # scale
        # [batch_size, input_seq_len, output_seq_len]
        head_size = tf.cast(tf.shape(K_layer)[-1], dtype=tf.float32)
        attention = tf.divide(attention, tf.sqrt(head_size))
        # mask
        # L670@bert_self_attention.py
        # [batch_size, input_seq_len, input_seq_len]
        attention = tf.nn.softmax(attention, dim=-1)
        # [batch_size, input_seq_len, output_seq_len]
        outputs = tf.matmul(attention, V_layer)
        return outputs
