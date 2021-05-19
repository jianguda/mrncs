from typing import Dict, Any, Union, Tuple

import tensorflow as tf

from .common import Common
from ..masked_seq_encoder import MaskedSeqEncoder
from ..utils.bert_self_attention import BertConfig, BertModel
from utils.tfutils import get_activation, write_to_feed_dict, pool_sequence_embedding


def __make_rnn_cell(cell_type: str,
                    hidden_size: int,
                    dropout_keep_rate: Union[float, tf.Tensor]=1.0,
                    recurrent_dropout_keep_rate: Union[float, tf.Tensor]=1.0) \
        -> tf.nn.rnn_cell.RNNCell:
    """
    Args:
        cell_type: "lstm", "gru", or 'rnn' (any casing)
        hidden_size: size for the underlying recurrent unit
        dropout_keep_rate: output-vector dropout prob
        recurrent_dropout_keep_rate:  state-vector dropout prob

    Returns:
        RNNCell of the desired type.
    """
    cell_type = cell_type.lower()
    if cell_type == 'lstm':
        cell = tf.nn.rnn_cell.LSTMCell(hidden_size)
    elif cell_type == 'gru':
        cell = tf.nn.rnn_cell.GRUCell(hidden_size)
    elif cell_type == 'rnn':
        cell = tf.nn.rnn_cell.BasicRNNCell(hidden_size)
    else:
        raise ValueError("Unknown RNN cell type '%s'!" % cell_type)

    return tf.contrib.rnn.DropoutWrapper(cell,
                                         output_keep_prob=dropout_keep_rate,
                                         state_keep_prob=recurrent_dropout_keep_rate)


def _make_deep_rnn_cell(num_layers: int,
                        cell_type: str,
                        hidden_size: int,
                        dropout_keep_rate: Union[float, tf.Tensor]=1.0,
                        recurrent_dropout_keep_rate: Union[float, tf.Tensor]=1.0) -> tf.nn.rnn_cell.RNNCell:
    """
    Args:
        num_layers: number of layers in result
        cell_type: "lstm" or "gru" (any casing)
        hidden_size: size for the underlying recurrent unit
        dropout_keep_rate: output-vector dropout prob
        recurrent_dropout_keep_rate: state-vector dropout prob

    Returns:
        (Multi)RNNCell of the desired type.
    """
    if num_layers == 1:
        return __make_rnn_cell(cell_type, hidden_size, dropout_keep_rate, recurrent_dropout_keep_rate)
    cells = [__make_rnn_cell(cell_type, hidden_size, dropout_keep_rate, recurrent_dropout_keep_rate)
             for _ in range(num_layers)]
    return tf.nn.rnn_cell.MultiRNNCell(cells)


class TreeSeqEncoder(MaskedSeqEncoder):
    @classmethod
    def get_default_hyperparameters(cls) -> Dict[str, Any]:
        encoder_hypers = {'nbow_pool_mode': 'weighted_mean',

                          '1dcnn_position_encoding': 'learned',  # One of {'none', 'learned'}
                          '1dcnn_layer_list': [128, 128, 128],
                          '1dcnn_kernel_width': [16, 16, 16],  # Has to have same length as 1dcnn_layer_list
                          '1dcnn_add_residual_connections': True,
                          '1dcnn_activation': 'tanh',
                          '1dcnn_pool_mode': 'weighted_mean',

                          'rnn_num_layers': 2,
                          'rnn_hidden_dim': 64,
                          'rnn_cell_type': 'LSTM',  # One of [LSTM, GRU, RNN]
                          'rnn_is_bidirectional': True,
                          'rnn_dropout_keep_rate': 0.8,
                          'rnn_recurrent_dropout_keep_rate': 1.0,
                          'rnn_pool_mode': 'weighted_mean',

                          'self_attention_activation': 'gelu',
                          'self_attention_hidden_size': 128,
                          'self_attention_intermediate_size': 16,  # 512
                          'self_attention_num_layers': 2,
                          'self_attention_num_heads': 2,  # 8
                          'self_attention_pool_mode': 'weighted_mean',
                          }
        hypers = super().get_default_hyperparameters()
        hypers.update(encoder_hypers)
        return hypers

    def __init__(self, label: str, hyperparameters: Dict[str, Any], metadata: Dict[str, Any]):
        super().__init__(label, hyperparameters, metadata)

    @property
    def output_representation_size(self):
        return self.get_hyper('self_attention_hidden_size')

    def _encode_with_rnn(self,
                         inputs: tf.Tensor,
                         input_lengths: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        cell_type = self.get_hyper('rnn_cell_type').lower()
        rnn_cell_fwd = _make_deep_rnn_cell(num_layers=self.get_hyper('rnn_num_layers'),
                                           cell_type=cell_type,
                                           hidden_size=self.get_hyper('rnn_hidden_dim'),
                                           dropout_keep_rate=self.placeholders['rnn_dropout_keep_rate'],
                                           recurrent_dropout_keep_rate=self.placeholders['rnn_recurrent_dropout_keep_rate'],
                                           )
        if not self.get_hyper('rnn_is_bidirectional'):
            (outputs, final_states) = tf.nn.dynamic_rnn(cell=rnn_cell_fwd,
                                                        inputs=inputs,
                                                        sequence_length=input_lengths,
                                                        dtype=tf.float32,
                                                        )

            if cell_type == 'lstm':
                final_state = tf.concat([tf.concat(layer_final_state, axis=-1)  # concat c & m of LSTM cell
                                         for layer_final_state in final_states],
                                        axis=-1)  # concat across layers
            elif cell_type in ['gru', 'rnn']:
                final_state = tf.concat(final_states, axis=-1)
            else:
                raise ValueError("Unknown RNN cell type '%s'!" % cell_type)
        else:
            rnn_cell_bwd = _make_deep_rnn_cell(num_layers=self.get_hyper('rnn_num_layers'),
                                               cell_type=cell_type,
                                               hidden_size=self.get_hyper('rnn_hidden_dim'),
                                               dropout_keep_rate=self.placeholders['rnn_dropout_keep_rate'],
                                               recurrent_dropout_keep_rate=self.placeholders['rnn_recurrent_dropout_keep_rate'],
                                               )

            (outputs, final_states) = tf.nn.bidirectional_dynamic_rnn(cell_fw=rnn_cell_fwd,
                                                                      cell_bw=rnn_cell_bwd,
                                                                      inputs=inputs,
                                                                      sequence_length=input_lengths,
                                                                      dtype=tf.float32,
                                                                      )
            # Merge fwd/bwd outputs:
            if cell_type == 'lstm':
                final_state = tf.concat([tf.concat([tf.concat(layer_final_state, axis=-1)  # concat c & m of LSTM cell
                                                    for layer_final_state in layer_final_states],
                                                   axis=-1)  # concat across layers
                                        for layer_final_states in final_states],
                                        axis=-1)  # concat fwd & bwd
            elif cell_type in ['gru', 'rnn']:
                final_state = tf.concat([tf.concat(layer_final_states, axis=-1)  # concat across layers
                                         for layer_final_states in final_states],
                                        axis=-1)  # concat fwd & bwd
            else:
                raise ValueError("Unknown RNN cell type '%s'!" % cell_type)
            outputs = tf.concat(outputs, axis=-1)  # concat fwd & bwd

        return final_state, outputs

    def __add_position_encoding(self, seq_inputs: tf.Tensor) -> tf.Tensor:
        position_encoding = self.get_hyper('1dcnn_position_encoding').lower()
        if position_encoding == 'none':
            return seq_inputs
        elif position_encoding == 'learned':
            position_embeddings = \
                tf.get_variable(name='position_embeddings',
                                initializer=tf.truncated_normal_initializer(stddev=0.02),
                                shape=[self.get_hyper('max_num_tokens'),
                                       self.get_hyper('token_embedding_size')],
                                )
            # Add batch dimension to position embeddings to make broadcasting work, then add:
            return seq_inputs + tf.expand_dims(position_embeddings, axis=0)
        else:
            raise ValueError("Unknown position encoding '%s'!" % position_encoding)

    def init_minibatch(self, batch_data: Dict[str, Any]) -> None:
        super().init_minibatch(batch_data)
        # batch_data['tokens'] = []
        batch_data['tokens_lengths'] = []

    def minibatch_to_feed_dict(self, batch_data: Dict[str, Any], feed_dict: Dict[tf.Tensor, Any], is_train: bool) -> None:
        super().minibatch_to_feed_dict(batch_data, feed_dict, is_train)
        feed_dict[self.placeholders['rnn_dropout_keep_rate']] = \
            self.get_hyper('rnn_dropout_keep_rate') if is_train else 1.0
        feed_dict[self.placeholders['rnn_recurrent_dropout_keep_rate']] = \
            self.get_hyper('rnn_recurrent_dropout_keep_rate') if is_train else 1.0

        # write_to_feed_dict(feed_dict, self.placeholders['tokens'], batch_data['tokens'])
        write_to_feed_dict(feed_dict, self.placeholders['tokens_lengths'], batch_data['tokens_lengths'])

    def make_model(self, is_train: bool = False) -> tf.Tensor:
        # it is meaningless to combine two models on same data
        # therefore, I always use _single_model for real
        for_real = True
        if for_real:
            # baseline models for tokens
            return self._single_model(is_train)
        else:
            # (nbow + rnn) for tokens
            return self._complex_model(is_train)

    def _single_model(self, is_train: bool = False) -> tf.Tensor:
        model = 'nbow'  # nbow, cnn, rnn, bert
        embedding = None
        with tf.variable_scope("tree_encoder"):
            self._make_placeholders()

            self.placeholders['tokens_lengths'] = \
                tf.placeholder(tf.int32, shape=[None], name='tokens_lengths')
            self.placeholders['rnn_dropout_keep_rate'] = \
                tf.placeholder(tf.float32, shape=[], name='rnn_dropout_keep_rate')
            self.placeholders['rnn_recurrent_dropout_keep_rate'] = \
                tf.placeholder(tf.float32, shape=[], name='rnn_recurrent_dropout_keep_rate')

            attention = False
            if model == 'nbow':
                seq_tokens_embeddings = self.embedding_layer(self.placeholders['tokens'])
                seq_token_mask = self.placeholders['tokens_mask']
                seq_token_lengths = tf.reduce_sum(seq_token_mask, axis=1)  # B

                if attention:
                    embedding = Common.yet_attention_layer(seq_tokens_embeddings)
                else:
                    embedding = pool_sequence_embedding(self.get_hyper('nbow_pool_mode').lower(),
                                                        sequence_token_embeddings=seq_tokens_embeddings,
                                                        sequence_lengths=seq_token_lengths,
                                                        sequence_token_masks=seq_token_mask)
            elif model == 'cnn':
                seq_tokens_embeddings = self.embedding_layer(self.placeholders['tokens'])
                seq_tokens_embeddings = self.__add_position_encoding(seq_tokens_embeddings)

                activation_fun = get_activation(self.get_hyper('1dcnn_activation'))
                current_embeddings = seq_tokens_embeddings
                num_filters_and_width = zip(self.get_hyper('1dcnn_layer_list'), self.get_hyper('1dcnn_kernel_width'))
                for (layer_idx, (num_filters, kernel_width)) in enumerate(num_filters_and_width):
                    next_embeddings = tf.layers.conv1d(
                        inputs=current_embeddings,
                        filters=num_filters,
                        kernel_size=kernel_width,
                        padding="same")

                    # Add residual connections past the first layer.
                    if self.get_hyper('1dcnn_add_residual_connections') and layer_idx > 0:
                        next_embeddings += current_embeddings

                    current_embeddings = activation_fun(next_embeddings)
                    current_embeddings = tf.nn.dropout(current_embeddings,
                                                       keep_prob=self.placeholders['dropout_keep_rate'])

                if attention:
                    embedding = Common.yet_attention_layer(current_embeddings)
                else:
                    seq_token_mask = self.placeholders['tokens_mask']
                    seq_token_lengths = tf.reduce_sum(seq_token_mask, axis=1)  # B
                    embedding = pool_sequence_embedding(self.get_hyper('1dcnn_pool_mode').lower(),
                                                        sequence_token_embeddings=current_embeddings,
                                                        sequence_lengths=seq_token_lengths,
                                                        sequence_token_masks=seq_token_mask)
            elif model == 'rnn':
                seq_tokens = self.placeholders['tokens']
                seq_tokens_embeddings = self.embedding_layer(seq_tokens)
                seq_tokens_lengths = self.placeholders['tokens_lengths']
                rnn_final_state, token_embeddings = self._encode_with_rnn(seq_tokens_embeddings, seq_tokens_lengths)

                output_pool_mode = self.get_hyper('rnn_pool_mode').lower()
                if output_pool_mode == 'rnn_final':
                    embedding = rnn_final_state
                else:
                    if attention:
                        embedding = Common.yet_attention_layer(token_embeddings)
                    else:
                        token_mask = tf.expand_dims(tf.range(tf.shape(seq_tokens)[1]), axis=0)  # 1 x T
                        token_mask = tf.tile(token_mask, multiples=(tf.shape(seq_tokens_lengths)[0], 1))  # B x T
                        token_mask = tf.cast(token_mask < tf.expand_dims(seq_tokens_lengths, axis=-1),
                                             dtype=tf.float32)  # B x T
                        embedding = pool_sequence_embedding(output_pool_mode,
                                                            sequence_token_embeddings=token_embeddings,
                                                            sequence_lengths=seq_tokens_lengths,
                                                            sequence_token_masks=token_mask)
            elif model == 'bert':
                config = BertConfig(vocab_size=self.get_hyper('token_vocab_size'),
                                    hidden_size=self.get_hyper('self_attention_hidden_size'),
                                    num_hidden_layers=self.get_hyper('self_attention_num_layers'),
                                    num_attention_heads=self.get_hyper('self_attention_num_heads'),
                                    intermediate_size=self.get_hyper('self_attention_intermediate_size'))

                model = BertModel(config=config,
                                  is_training=is_train,
                                  input_ids=self.placeholders['tokens'],
                                  input_mask=self.placeholders['tokens_mask'],
                                  use_one_hot_embeddings=False)

                output_pool_mode = self.get_hyper('self_attention_pool_mode').lower()
                if output_pool_mode == 'bert':
                    embedding = model.get_pooled_output()
                else:
                    seq_token_embeddings = model.get_sequence_output()
                    # only when it is not pooled out, then we consider attention
                    if attention:
                        embedding = Common.yet_attention_layer(seq_token_embeddings)
                    else:
                        seq_token_masks = self.placeholders['tokens_mask']
                        seq_token_lengths = tf.reduce_sum(seq_token_masks, axis=1)  # B
                        embedding = pool_sequence_embedding(output_pool_mode,
                                                            sequence_token_embeddings=seq_token_embeddings,
                                                            sequence_lengths=seq_token_lengths,
                                                            sequence_token_masks=seq_token_masks)
            else:
                raise ValueError('Undefined Config')
            return embedding

    def _complex_model(self, is_train: bool = False) -> tf.Tensor:
        embeddings = []
        with tf.variable_scope("tree_encoder"):
            self._make_placeholders()

            self.placeholders['tokens_lengths'] = \
                tf.placeholder(tf.int32, shape=[None], name='tokens_lengths')
            self.placeholders['rnn_dropout_keep_rate'] = \
                tf.placeholder(tf.float32, shape=[], name='rnn_dropout_keep_rate')
            self.placeholders['rnn_recurrent_dropout_keep_rate'] = \
                tf.placeholder(tf.float32, shape=[], name='rnn_recurrent_dropout_keep_rate')

            common_flag = True
            models = ['nbow', 'rnn']  # nbow, cnn, rnn, bert
            if 'nbow' in models and 'rnn' in models:
                seq_tokens = self.placeholders['tokens']
                seq_tokens_embeddings = self.embedding_layer(seq_tokens)
                common_flag = False
            if 'nbow' in models:
                if common_flag:
                    seq_tokens_embeddings = self.embedding_layer(self.placeholders['tokens'])
                seq_token_mask = self.placeholders['tokens_mask']
                seq_token_lengths = tf.reduce_sum(seq_token_mask, axis=1)  # B

                embedding = pool_sequence_embedding(self.get_hyper('nbow_pool_mode').lower(),
                                                    sequence_token_embeddings=seq_tokens_embeddings,
                                                    sequence_lengths=seq_token_lengths,
                                                    sequence_token_masks=seq_token_mask)
                embeddings.append(embedding)
            if 'cnn' in models:
                if common_flag:
                    seq_tokens_embeddings = self.embedding_layer(self.placeholders['tokens'])
                seq_tokens_embeddings = self.__add_position_encoding(seq_tokens_embeddings)

                activation_fun = get_activation(self.get_hyper('1dcnn_activation'))
                current_embeddings = seq_tokens_embeddings
                num_filters_and_width = zip(self.get_hyper('1dcnn_layer_list'), self.get_hyper('1dcnn_kernel_width'))
                for (layer_idx, (num_filters, kernel_width)) in enumerate(num_filters_and_width):
                    next_embeddings = tf.layers.conv1d(
                        inputs=current_embeddings,
                        filters=num_filters,
                        kernel_size=kernel_width,
                        padding="same")

                    # Add residual connections past the first layer.
                    if self.get_hyper('1dcnn_add_residual_connections') and layer_idx > 0:
                        next_embeddings += current_embeddings

                    current_embeddings = activation_fun(next_embeddings)
                    current_embeddings = tf.nn.dropout(current_embeddings,
                                                       keep_prob=self.placeholders['dropout_keep_rate'])

                seq_token_mask = self.placeholders['tokens_mask']
                seq_token_lengths = tf.reduce_sum(seq_token_mask, axis=1)  # B
                embedding = pool_sequence_embedding(self.get_hyper('1dcnn_pool_mode').lower(),
                                                    sequence_token_embeddings=current_embeddings,
                                                    sequence_lengths=seq_token_lengths,
                                                    sequence_token_masks=seq_token_mask)
                embeddings.append(embedding)
            if 'rnn' in models:
                if common_flag:
                    seq_tokens = self.placeholders['tokens']
                    seq_tokens_embeddings = self.embedding_layer(seq_tokens)
                seq_tokens_lengths = self.placeholders['tokens_lengths']
                rnn_final_state, token_embeddings = self._encode_with_rnn(seq_tokens_embeddings, seq_tokens_lengths)

                output_pool_mode = self.get_hyper('rnn_pool_mode').lower()
                if output_pool_mode == 'rnn_final':
                    embedding = rnn_final_state
                else:
                    token_mask = tf.expand_dims(tf.range(tf.shape(seq_tokens)[1]), axis=0)  # 1 x T
                    token_mask = tf.tile(token_mask, multiples=(tf.shape(seq_tokens_lengths)[0], 1))  # B x T
                    token_mask = tf.cast(token_mask < tf.expand_dims(seq_tokens_lengths, axis=-1),
                                         dtype=tf.float32)  # B x T
                    embedding = pool_sequence_embedding(output_pool_mode,
                                                        sequence_token_embeddings=token_embeddings,
                                                        sequence_lengths=seq_tokens_lengths,
                                                        sequence_token_masks=token_mask)
                embeddings.append(embedding)
            if 'bert' in models:
                config = BertConfig(vocab_size=self.get_hyper('token_vocab_size'),
                                    hidden_size=self.get_hyper('self_attention_hidden_size'),
                                    num_hidden_layers=self.get_hyper('self_attention_num_layers'),
                                    num_attention_heads=self.get_hyper('self_attention_num_heads'),
                                    intermediate_size=self.get_hyper('self_attention_intermediate_size'))

                model = BertModel(config=config,
                                  is_training=is_train,
                                  input_ids=self.placeholders['tokens'],
                                  input_mask=self.placeholders['tokens_mask'],
                                  use_one_hot_embeddings=False)

                output_pool_mode = self.get_hyper('self_attention_pool_mode').lower()
                if output_pool_mode == 'bert':
                    embedding = model.get_pooled_output()
                else:
                    seq_token_embeddings = model.get_sequence_output()
                    seq_token_masks = self.placeholders['tokens_mask']
                    seq_token_lengths = tf.reduce_sum(seq_token_masks, axis=1)  # B
                    embedding = pool_sequence_embedding(output_pool_mode,
                                                        sequence_token_embeddings=seq_token_embeddings,
                                                        sequence_lengths=seq_token_lengths,
                                                        sequence_token_masks=seq_token_masks)
                embeddings.append(embedding)

            embeddings = tf.concat(embeddings, axis=-1)
            attention = False
            if attention:
                embeddings = Common.self_attention_layer(embeddings)
            # "concat one-hot" is equal to "accumulate embedding"
            # [v1^T, v2^T, v3^T] * W = [v1^T, v2^T, v3^T]*[w1, w2, w3]^T = v1^T*w1+v2^T*w2+v3^T*w3
            print('*@' * 16)
            print(embeddings)
            print(tf.shape(embeddings))
            return tf.reduce_sum(embeddings, axis=0)
            # return tf.reduce_mean(embeddings, axis=0)
