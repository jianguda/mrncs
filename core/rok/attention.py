from tensorflow.keras import backend
from tensorflow.keras import layers


class SelfAttention(layers.Layer):
    def __init__(self, hidden_dim, output_dim, **kwargs):
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        super().__init__(**kwargs)

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'hidden_dim': self.hidden_dim,
            'output_dim': self.output_dim,
        })
        return config

    def build(self, input_shape):
        self.WQ = self.add_weight(name='WQ',
                                  shape=(input_shape[-1], self.hidden_dim),
                                  initializer='uniform',
                                  trainable=True)
        self.WK = self.add_weight(name='WK',
                                  shape=(input_shape[-1], self.hidden_dim),
                                  initializer='uniform',
                                  trainable=True)
        self.WV = self.add_weight(name='WV',
                                  shape=(input_shape[-1], self.output_dim),
                                  initializer='uniform',
                                  trainable=True)
        super().build(input_shape)

    def call(self, inputs, **kwargs):
        # print(tf.shape(inputs), inputs.shape)
        WQ = backend.dot(inputs, self.WQ)
        WK = backend.dot(inputs, self.WK)
        WV = backend.dot(inputs, self.WV)
        WK = backend.permute_dimensions(WK, (0, 2, 1))
        QK = backend.batch_dot(WQ, WK)
        # tip: hidden_dim = math.sqrt(input_shape[0])
        QK = QK / (self.hidden_dim ** 0.5)
        QK = backend.softmax(QK)
        outputs = backend.batch_dot(QK, WV)
        return outputs

    def compute_output_shape(self, input_shape):
        return input_shape[0], input_shape[1], self.output_dim
