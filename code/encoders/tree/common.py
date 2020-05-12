import string

import nltk
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
import tensorflow as tf

nltk.download('stopwords')


class Common:
    # this implementation is better for single model
    @staticmethod
    def yet_attention_layer(embedding):
        # (B,T,D)
        # [batch_size, seq_length, hidden_size]
        hidden_size = embedding.shape[2].value  # D value - hidden size
        attention_size = 128
        initializer = tf.random_normal_initializer(stddev=0.1)

        # Trainable parameters
        w_omega = tf.get_variable(name="w_omega", shape=[hidden_size, attention_size], initializer=initializer)
        b_omega = tf.get_variable(name="b_omega", shape=[attention_size], initializer=initializer)
        u_omega = tf.get_variable(name="u_omega", shape=[attention_size], initializer=initializer)

        with tf.name_scope('att'):
            # Applying fully connected layer with non-linear activation to each of the B*T timestamps;
            #  the shape of `v` is (B,T,D)*(D,A)=(B,T,A), where A=attention_size
            v = tf.tanh(tf.tensordot(embedding, w_omega, axes=1) + b_omega)

        # For each of the timestamps its vector of size A from `v` is reduced with `u` vector
        vu = tf.tensordot(v, u_omega, axes=1, name='vu')  # (B,T) shape
        alphas = tf.nn.softmax(vu, name='alphas')  # (B,T) shape

        # Output of (Bi-)RNN is reduced with attention vector; the result has (B,D) shape
        output = tf.reduce_sum(embedding * tf.expand_dims(alphas, -1), 1)
        return output

    # we use this implementation for multi-modal
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

    # JGD
    #  for training & testing: load_metadata_from_sample & load_data_from_sample @model.py
    #  for predicting : load_data_from_sample @model.py
    #  with regard to encoders:
    #  load_data_from_sample & load_metadata_from_sample
    #  @leaf_encoder.py @path_encoder.py @seq_encoder.py
    #  1. remove stop-words, to make sure phrase query -> keyword query
    #  2. enable sub-tokens, ...
    @staticmethod
    def preprocessing_code(tokens):
        backup = [token.lower() for token in tokens]
        tokens = [token.lower() for token in tokens]
        # >>> from keyword import kwlist
        # >>> print(kwlist)
        keywords4python = [
            'False', 'None', 'True', 'and', 'as', 'assert', 'async', 'await', 'break', 'class', 'continue', 'def',
            'del', 'elif', 'else', 'except', 'finally', 'for', 'from', 'global', 'if', 'import', 'in', 'is', 'lambda',
            'nonlocal', 'not', 'or', 'pass', 'raise', 'return', 'try', 'while', 'with', 'yield',
        ]
        keywords4python = [token.lower() for token in keywords4python]
        # >>> import builtins
        # >>> dir(builtins)
        builtins4python = [
            'ArithmeticError', 'AssertionError', 'AttributeError', 'BaseException', 'BlockingIOError',
            'BrokenPipeError', 'BufferError', 'BytesWarning', 'ChildProcessError', 'ConnectionAbortedError',
            'ConnectionError', 'ConnectionRefusedError', 'ConnectionResetError', 'DeprecationWarning', 'EOFError',
            'Ellipsis', 'EnvironmentError', 'Exception', 'False', 'FileExistsError', 'FileNotFoundError',
            'FloatingPointError', 'FutureWarning', 'GeneratorExit', 'IOError', 'ImportError', 'ImportWarning',
            'IndentationError', 'IndexError', 'InterruptedError', 'IsADirectoryError', 'KeyError', 'KeyboardInterrupt',
            'LookupError', 'MemoryError', 'ModuleNotFoundError', 'NameError', 'None', 'NotADirectoryError',
            'NotImplemented', 'NotImplementedError', 'OSError', 'OverflowError', 'PendingDeprecationWarning',
            'PermissionError', 'ProcessLookupError', 'RecursionError', 'ReferenceError', 'ResourceWarning',
            'RuntimeError', 'RuntimeWarning', 'StopAsyncIteration', 'StopIteration', 'SyntaxError', 'SyntaxWarning',
            'SystemError', 'SystemExit', 'TabError', 'TimeoutError', 'True', 'TypeError', 'UnboundLocalError',
            'UnicodeDecodeError', 'UnicodeEncodeError', 'UnicodeError', 'UnicodeTranslateError', 'UnicodeWarning',
            'UserWarning', 'ValueError', 'Warning', 'WindowsError', 'ZeroDivisionError', '_', '__build_class__',
            '__debug__', '__doc__', '__import__', '__loader__', '__name__', '__package__', '__spec__', 'abs', 'all',
            'any', 'ascii', 'bin', 'bool', 'bytearray', 'bytes', 'callable', 'chr', 'classmethod', 'compile', 'complex',
            'copyright', 'credits', 'delattr', 'dict', 'dir', 'divmod', 'enumerate', 'eval', 'exec', 'exit', 'filter',
            'float', 'format', 'frozenset', 'getattr', 'globals', 'hasattr', 'hash', 'help', 'hex', 'id', 'input',
            'int', 'isinstance', 'issubclass', 'iter', 'len', 'license', 'list', 'locals', 'map', 'max', 'memoryview',
            'min', 'next', 'object', 'oct', 'open', 'ord', 'pow', 'print', 'property', 'quit', 'range', 'repr',
            'reversed', 'round', 'set', 'setattr', 'slice', 'sorted', 'staticmethod', 'str', 'sum', 'super', 'tuple',
            'type', 'vars', 'zip'
        ]
        builtins4python = [token.lower() for token in builtins4python]
        # >>> characters
        characters4python = [
            '(', ')', '[', ']', '{', '}', ',', '.', ':', '+', '-', '*', '/', '%', '=', '!', '>', '<',
            '\'', '\"', '\\n', '\\t'
        ]
        strategy = 'convert'
        # strategy = 'discard'
        if strategy == 'convert':
            # transform language-specific stopwords
            tokens = [token if token not in keywords4python else 'keyword' for token in tokens]
            # tokens = [token if token not in builtins4python else 'builtin' for token in tokens]
            tokens = [token if token not in characters4python else 'character' for token in tokens]
        elif strategy == 'discard':
            # remove language-specific stopwords
            tokens = [token for token in tokens if token not in keywords4python]
            # tokens = [token for token in tokens if token not in builtins4python]
            tokens = [token for token in tokens if token not in characters4python]
        return tokens if tokens else backup

    @staticmethod
    def preprocessing_query(tokens):
        backup = [token.lower() for token in tokens]
        tokens = [token.lower() for token in tokens]
        # remove punctuations and non-alphas
        # tokens = [token for token in tokens if len(token) > 1]
        # tokens = [token for token in tokens if len(token) > 2]
        # tokens = [token for token in tokens if len(token) > 3]
        # tokens = [token for token in tokens if not token.isdigit()]
        # tokens = [token for token in tokens if token not in string.punctuation]
        # tokens = [token for token in tokens if token.isalpha()]
        # remove stop-words
        stop_words = stopwords.words('english')
        tokens = [token for token in tokens if token not in stop_words]
        # stemming
        # https://easyai.tech/ai-definition/stemming-lemmatisation/
        # snowball_stemmer = SnowballStemmer('english')
        # tokens = [snowball_stemmer.stem(token) for token in tokens]
        # deduplication
        # tokens = list(set(tokens))
        return tokens if tokens else backup
