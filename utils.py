from tensorflow import keras
import tensorflow as tf
from tensorflow.python.keras import backend as K
import params_flow as pf
from params_flow.activations import gelu
from params_flow import LayerNormalization


class Layer(pf.Layer):
    """ Common abstract base layer for all BERT layers. """
    class Params(pf.Layer.Params):
        initializer_range = 0.02

    def create_initializer(self):
        return tf.keras.initializers.TruncatedNormal(stddev=self.params.initializer_range)

    @staticmethod
    def get_activation(activation_string):
        if not isinstance(activation_string, str):
            return activation_string
        if not activation_string:
            return None

        act = activation_string.lower()
        if act == "linear":
            return None
        elif act == "relu":
            return tf.nn.relu
        elif act == "gelu":
            return gelu
        elif act == "tanh":
            return tf.tanh
        elif act == "sigmoid":
            return tf.sigmoid
        else:
            raise ValueError("Unsupported activation: %s" % act)

class MyAttention(keras.layers.Layer):
    """ Keras layer for Mutiply a Tensor to be the same shape as another Tensor.
    """

    def __init__(self, num_heads=None, size_per_head=None, **kwargs):
        self.num_heads = num_heads
        self.size_per_head = size_per_head
        super(MyAttention, self).__init__(**kwargs)
    
    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'num_heads': self.num_heads,
            'size_per_head': self.size_per_head 
        })
        return config

    def call(self, inputs,training=None,**kwargs):
        value, attention_probs = inputs
        input_shape = tf.shape(input=value)
        batch_size, from_seq_len = input_shape[0], input_shape[1]
        to_seq_len = from_seq_len
        value = tf.reshape(value, [batch_size, to_seq_len,
                                   self.num_heads, self.size_per_head])
        value = tf.transpose(a=value, perm=[0, 2, 1, 3])  # [B, N, T, H]
        context_layer = tf.matmul(attention_probs, value)  # [B, N, F, H]
        context_layer = tf.transpose(a=context_layer, perm=[0, 2, 1, 3])  # [B, F, N, H]
        output_shape = [batch_size, from_seq_len,
                        self.num_heads * self.size_per_head]
        context_layer = tf.reshape(context_layer, output_shape)
        return context_layer

    def compute_output_shape(self, input_shape):
        from_shape = input_shape
        output_shape = [from_shape[0], from_shape[1], self.num_heads * self.size_per_head]
        return output_shape  # [B, F, N*H], [B, F, T]


class PositionEmbeddingLayer(Layer):
    """ Keras layer for Position Embedding.
    """
    class Params(Layer.Params):
        max_position_embeddings  = 512
        hidden_size              = 190

    # noinspection PyUnusedLocal
    def _construct(self, **kwargs):
        super()._construct(**kwargs)
        self.embedding_table = None

    # noinspection PyAttributeOutsideInit
    def build(self, input_shape):
        # input_shape: () of seq_len
        if input_shape is not None:
            assert input_shape.ndims == 0
            self.input_spec = keras.layers.InputSpec(shape=input_shape, dtype='int32')
        else:
            self.input_spec = keras.layers.InputSpec(shape=(), dtype='int32')

        self.embedding_table = self.add_weight(name="embeddings",
                                               dtype=K.floatx(),
                                               shape=[self.params.max_position_embeddings, self.params.hidden_size],
                                               initializer=self.create_initializer())
        super(PositionEmbeddingLayer, self).build(input_shape)

    # noinspection PyUnusedLocal
    def call(self, inputs, **kwargs):
        # just return the embedding after verifying
        # that seq_len is less than max_position_embeddings

        seq_len = inputs

        assert_op = tf.compat.v2.debugging.assert_less_equal(seq_len, self.params.max_position_embeddings)

        with tf.control_dependencies([assert_op]):
            # slice to seq_len
            full_position_embeddings = tf.slice(self.embedding_table,
                                                [0, 0],
                                                [seq_len, -1])
        output = full_position_embeddings
        return output



class EmbeddingsProjector(Layer):
    class Params(Layer.Params):
        hidden_size                  = 190
        embedding_size               = None   # None for BERT, not None for ALBERT
        project_embeddings_with_bias = True   # in ALBERT - True for Google, False for brightmart/albert_zh

    # noinspection PyUnusedLocal
    def _construct(self, **kwargs):
        super()._construct(**kwargs)
        self.projector_layer      = None   # for ALBERT
        self.projector_bias_layer = None   # for ALBERT

    def build(self, input_shape):
        emb_shape = input_shape
        self.input_spec = keras.layers.InputSpec(shape=emb_shape)
        assert emb_shape[-1] == self.params.embedding_size

        # ALBERT word embeddings projection
        self.projector_layer = self.add_weight(name="projector",
                                               shape=[self.params.embedding_size,
                                                      self.params.hidden_size],
                                               dtype=K.floatx())
        if self.params.project_embeddings_with_bias:
            self.projector_bias_layer = self.add_weight(name="bias",
                                                        shape=[self.params.hidden_size],
                                                        dtype=K.floatx())
        super(EmbeddingsProjector, self).build(input_shape)

    def call(self, inputs, **kwargs):
        input_embedding = inputs
        assert input_embedding.shape[-1] == self.params.embedding_size

        # ALBERT: project embedding to hidden_size
        output = tf.matmul(input_embedding, self.projector_layer)
        if self.projector_bias_layer is not None:
            output = tf.add(output, self.projector_bias_layer)

        return output


class BertEmbeddingsLayer(Layer):
    class Params(PositionEmbeddingLayer.Params,
                 EmbeddingsProjector.Params):
        vocab_size               = None
        use_token_type           = None
        use_position_embeddings  = True
        token_type_vocab_size    = 0 # segment_ids类别 [0,1]
        hidden_size              = 190
        hidden_dropout           = 0.1

        extra_tokens_vocab_size  = None  # size of the extra (task specific) token vocabulary (using negative token ids)

        #
        # ALBERT support - set embedding_size (or None for BERT)
        #
        embedding_size               = None   # None for BERT, not None for ALBERT
        project_embeddings_with_bias = False   # in ALBERT - True for Google, False for brightmart/albert_zh
        project_position_embeddings  = False   # in ALEBRT - True for Google, False for brightmart/albert_zh

        mask_zero                    = False

    # noinspection PyUnusedLocal
    def _construct(self, **kwargs):
        super()._construct(**kwargs)
        self.word_embeddings_layer       = None
        self.extra_word_embeddings_layer = None   # for task specific tokens (negative token ids)
        self.token_type_embeddings_layer = None
        self.position_embeddings_layer   = None
        self.word_embeddings_projector_layer = None   # for ALBERT
        self.layer_norm_layer = None
        self.dropout_layer    = None

        self.support_masking = self.params.mask_zero

    # noinspection PyAttributeOutsideInit
    def build(self, input_shape):
        if isinstance(input_shape, list):
            assert len(input_shape) == 2
            input_ids_shape, token_type_ids_shape = input_shape
            self.input_spec = [keras.layers.InputSpec(shape=input_ids_shape),
                               keras.layers.InputSpec(shape=token_type_ids_shape)]
        else:
            input_ids_shape = input_shape
            self.input_spec = keras.layers.InputSpec(shape=input_ids_shape)

        # use either hidden_size for BERT or embedding_size for ALBERT
        embedding_size = self.params.hidden_size if self.params.embedding_size is None else self.params.embedding_size

        self.word_embeddings_layer = keras.layers.Embedding(
            input_dim=self.params.vocab_size,
            output_dim=embedding_size,
            mask_zero=self.params.mask_zero,
            name="word_embeddings"
        )
        if self.params.extra_tokens_vocab_size is not None:
            self.extra_word_embeddings_layer = keras.layers.Embedding(
                input_dim=self.params.extra_tokens_vocab_size + 1,  # +1 is for a <pad>/0 vector
                output_dim=embedding_size,
                mask_zero=self.params.mask_zero,
                embeddings_initializer=self.create_initializer(),
                name="extra_word_embeddings"
            )

        # ALBERT word embeddings projection
        if self.params.embedding_size is not None:
            self.word_embeddings_projector_layer = EmbeddingsProjector.from_params(
                self.params, name="word_embeddings_projector")

        position_embedding_size = embedding_size if self.params.project_position_embeddings else self.params.hidden_size

        if self.params.use_token_type:
            self.token_type_embeddings_layer = keras.layers.Embedding(
                input_dim=self.params.token_type_vocab_size,
                output_dim=position_embedding_size,
                mask_zero=False,
                name="token_type_embeddings"
            )
        if self.params.use_position_embeddings:
            self.position_embeddings_layer = PositionEmbeddingLayer.from_params(
                self.params,
                name="position_embeddings",
                hidden_size=position_embedding_size
            )

        self.layer_norm_layer = pf.LayerNormalization(name="LayerNorm")
        self.dropout_layer    = keras.layers.Dropout(rate=self.params.hidden_dropout)

        super(BertEmbeddingsLayer, self).build(input_shape)

    def call(self, inputs, mask=None, training=None):
        if isinstance(inputs, list):
            assert 2 == len(inputs), "Expecting inputs to be a [input_ids, token_type_ids] list"
            input_ids, token_type_ids = inputs
        else:
            input_ids      = inputs
            token_type_ids = None

        input_ids = tf.cast(input_ids, dtype=tf.float32)

        if self.extra_word_embeddings_layer is not None:
            token_mask   = tf.cast(tf.greater_equal(input_ids, 0), tf.int32)
            extra_mask   = tf.cast(tf.less(input_ids, 0), tf.int32)
            token_ids    = token_mask * input_ids
            extra_tokens = extra_mask * (-input_ids)
            token_output = self.word_embeddings_layer(token_ids)
            extra_output = self.extra_word_embeddings_layer(extra_tokens)
            embedding_output = tf.add(token_output,
                                      extra_output * tf.expand_dims(tf.cast(extra_mask, K.floatx()), axis=-1))
        else:
            embedding_output = input_ids # 这里做了相应的修改，去掉了word_embedding
            # embedding_output = self.word_embeddings_layer(input_ids)

        # ALBERT: for brightmart/albert_zh weights - project only token embeddings
        if not self.params.project_position_embeddings:
            if self.word_embeddings_projector_layer:
                embedding_output = self.word_embeddings_projector_layer(embedding_output)

        if token_type_ids is not None:
            token_type_ids    = tf.cast(token_type_ids, dtype=tf.int32)
            embedding_output += self.token_type_embeddings_layer(token_type_ids)

        if self.position_embeddings_layer is not None:
            seq_len  = input_ids.shape.as_list()[1]
            emb_size = embedding_output.shape[-1]

            pos_embeddings = self.position_embeddings_layer(seq_len)
            # broadcast over all dimension except the last two [..., seq_len, width]
            broadcast_shape = [1] * (embedding_output.shape.ndims - 2) + [seq_len, emb_size]
            embedding_output += tf.reshape(pos_embeddings, broadcast_shape)

        embedding_output = self.layer_norm_layer(embedding_output)
        embedding_output = self.dropout_layer(embedding_output, training=training)

        # ALBERT: for google-research/albert weights - project all embeddings
        if self.params.project_position_embeddings:
            if self.word_embeddings_projector_layer:
                embedding_output = self.word_embeddings_projector_layer(embedding_output)

        return embedding_output   # [B, seq_len, hidden_size]

    def compute_mask(self, inputs, mask=None):
        if isinstance(inputs, list):
            assert 2 == len(inputs), "Expecting inputs to be a [input_ids, token_type_ids] list"
            input_ids, token_type_ids = inputs
        else:
            input_ids      = inputs
            token_type_ids = None

        if not self.support_masking:
            return None

        return tf.not_equal(input_ids, 0)


class AttentionLayer(Layer):
    class Params(Layer.Params):
        num_heads         = None
        size_per_head     = None
        initializer_range = 0.02
        query_activation  = None
        key_activation    = None
        value_activation  = None
        attention_dropout = 0.1
        negative_infinity = -10000.0  # used for attention scores before softmax


    @staticmethod
    def create_attention_mask(from_shape, input_mask):
        """
        Creates 3D attention.
        :param from_shape:  [batch_size, from_seq_len, ...]
        :param input_mask:  [batch_size, seq_len]
        :return: [batch_size, from_seq_len, seq_len]
        """

        mask = tf.cast(tf.expand_dims(input_mask, axis=1), tf.float32)                   # [B, 1, T]
        ones = tf.expand_dims(tf.ones(shape=from_shape[:2], dtype=tf.float32), axis=-1)  # [B, F, 1]
        mask = ones * mask  # broadcast along two dimensions

        return mask  # [B, F, T]

    def _construct(self, **kwargs):
        super()._construct(**kwargs)
        self.query_activation = self.params.query_activation
        self.key_activation   = self.params.key_activation
        self.value_activation = self.params.value_activation

        self.query_layer = None
        self.key_layer   = None
        self.value_layer = None

        self.supports_masking = True

    # noinspection PyAttributeOutsideInit
    def build(self, input_shape):
        self.input_spec = keras.layers.InputSpec(shape=input_shape)

        dense_units = self.params.num_heads * self.params.size_per_head  # N*H
        #
        # B, F, T, N, H - batch, from_seq_len, to_seq_len, num_heads, size_per_head
        #
        self.query_layer = keras.layers.Dense(units=dense_units, activation=self.query_activation,
                                              kernel_initializer=self.create_initializer(),
                                              name="query")
        self.key_layer   = keras.layers.Dense(units=dense_units, activation=self.key_activation,
                                              kernel_initializer=self.create_initializer(),
                                              name="key")
        self.value_layer = keras.layers.Dense(units=dense_units, activation=self.value_activation,
                                              kernel_initializer=self.create_initializer(),
                                              name="value")
        self.dropout_layer = keras.layers.Dropout(self.params.attention_dropout)

        super(AttentionLayer, self).build(input_shape)

    def compute_output_shape(self, input_shape):
        from_shape = input_shape

        # from_shape         # [B, F, W]   [batch_size, from_seq_length, from_width] from_with = embedding_len(应该)
        # input_mask_shape   # [B, F]

        output_shape = [from_shape[0], from_shape[1], self.params.num_heads * self.params.size_per_head]
        attention_shape = [from_shape[0],self.params.num_heads,from_shape[1],from_shape[1]]
        return [output_shape,attention_shape]  # [B, F, N*H], [B, F, T]

    # noinspection PyUnusedLocal
    def call(self, inputs, mask=None, training=None, **kwargs):
        from_tensor = inputs
        to_tensor   = inputs
        if mask is None:
            sh = self.get_shape_list(from_tensor)
            mask = tf.ones(sh[:2], dtype=tf.int32)
        attention_mask = AttentionLayer.create_attention_mask(tf.shape(input=from_tensor), mask)

        #  from_tensor shape - [batch_size, from_seq_length, from_width]
        input_shape  = tf.shape(input=from_tensor)
        batch_size, from_seq_len, from_width = input_shape[0], input_shape[1], input_shape[2]
        to_seq_len = from_seq_len

        # [B, F, N*H] -> [B, N, F, H]
        def transpose_for_scores(input_tensor, seq_len):
            output_shape = [batch_size, seq_len,
                            self.params.num_heads, self.params.size_per_head]
            output_tensor = K.reshape(input_tensor, output_shape)
            return tf.transpose(a=output_tensor, perm=[0, 2, 1, 3])  # [B,N,F,H]

        query = self.query_layer(from_tensor)  # [B,F, N*H] [batch_size, from_seq_len, N*H]
        key   = self.key_layer(to_tensor)      # [B,T, N*H]
        value = self.key_layer(to_tensor)    # [B,T, N*H]

        query = transpose_for_scores(query, from_seq_len)           # [B, N, F, H]
        key   = transpose_for_scores(key,   to_seq_len)             # [B, N, T, H]

        attention_scores = tf.matmul(query, key, transpose_b=True)  # [B, N, F, T]
        attention_scores = attention_scores / tf.sqrt(float(self.params.size_per_head))

        if attention_mask is not None:
            attention_mask = tf.expand_dims(attention_mask, axis=1)  # [B, 1, F, T]
            # {1, 0} -> {0.0, -inf}
            adder = (1.0 - tf.cast(attention_mask, tf.float32)) * self.params.negative_infinity
            attention_scores = tf.add(attention_scores, adder)  # adding to softmax -> its like removing them entirely

        # scores to probabilities
        attention_probs = tf.nn.softmax(attention_scores)           # [B, N, F, T]

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout_layer(attention_probs,
                                             training=training)    # [B, N, F, T]

        # [B,T,N,H]
        value = tf.reshape(value, [batch_size, to_seq_len,
                                   self.params.num_heads, self.params.size_per_head])
        value = tf.transpose(a=value, perm=[0, 2, 1, 3])                                # [B, N, T, H]

        context_layer = tf.matmul(attention_probs, value)                               # [B, N, F, H]
        context_layer = tf.transpose(a=context_layer, perm=[0, 2, 1, 3])                # [B, F, N, H]

        output_shape = [batch_size, from_seq_len,
                        self.params.num_heads * self.params.size_per_head]
        context_layer = tf.reshape(context_layer, output_shape)
        return [context_layer, attention_scores]                                         # [B, F, N*H]

    # noinspection PyUnusedLocal
    def compute_mask(self, inputs, mask=None):
        return mask   # [B, F]
    
    
class Attention_SV(Layer):
    class Params(Layer.Params):
        num_heads = None
        size_per_head = None
        initializer_range = 0.02
        score = True
        query_activation = None
        key_activation = None
        value_activation = None
        attention_dropout = 0.1
        negative_infinity = -10000.0  # used for attention scores before softmax

    @staticmethod
    def create_attention_mask(from_shape, input_mask):
        """
        Creates 3D attention.
        :param from_shape:  [batch_size, from_seq_len, ...]
        :param input_mask:  [batch_size, seq_len]
        :return: [batch_size, from_seq_len, seq_len]
        """

        mask = tf.cast(tf.expand_dims(input_mask, axis=1), tf.float32)  # [B, 1, T]
        ones = tf.expand_dims(tf.ones(shape=from_shape[:2], dtype=tf.float32), axis=-1)  # [B, F, 1]
        mask = ones * mask  # broadcast along two dimensions

        return mask  # [B, F, T]

    def _construct(self, **kwargs):
        super()._construct(**kwargs)
        self.query_activation = self.params.query_activation
        self.key_activation = self.params.key_activation
        self.value_activation = self.params.value_activation

        self.query_layer = None
        self.key_layer = None
        self.value_layer = None

        self.supports_masking = True

    # noinspection PyAttributeOutsideInit
    def build(self, input_shape):
        self.input_spec = keras.layers.InputSpec(shape=input_shape)

        dense_units = self.params.num_heads * self.params.size_per_head  # N*H
        #
        # B, F, T, N, H - batch, from_seq_len, to_seq_len, num_heads, size_per_head
        #
        self.query_layer = keras.layers.Dense(units=dense_units, activation=self.query_activation,
                                              kernel_initializer=self.create_initializer(),
                                              name="query")
        self.key_layer = keras.layers.Dense(units=dense_units, activation=self.key_activation,
                                            kernel_initializer=self.create_initializer(),
                                            name="key")
        self.value_layer = keras.layers.Dense(units=dense_units, activation=self.value_activation,
                                              kernel_initializer=self.create_initializer(),
                                              name="value")
        self.dropout_layer = keras.layers.Dropout(self.params.attention_dropout)

        super(Attention_SV, self).build(input_shape)

    def compute_output_shape(self, input_shape):

        # from_shape         # [B, F, W]   [batch_size, from_seq_length, from_width] from_with = embedding_len(应该)
        # input_mask_shape   # [B, F]
        value_shape = input_shape
        attention_shape = [input_shape[0],self.params.num_heads,input_shape[1],input_shape[1]]
        if self.params.score == True:
            return [value_shape,attention_shape]  # [B, F, N*H], [B, F, T]
        else:
            return value_shape

    # noinspection PyUnusedLocal
    def call(self, inputs, mask=None, training=None, **kwargs):
        from_tensor = inputs
        to_tensor   = inputs
        if mask is None:
            sh = self.get_shape_list(from_tensor)
            mask = tf.ones(sh[:2], dtype=tf.int32)
        attention_mask = AttentionLayer.create_attention_mask(tf.shape(input=from_tensor), mask)

        #  from_tensor shape - [batch_size, from_seq_length, from_width]
        input_shape  = tf.shape(input=from_tensor)
        batch_size, from_seq_len, from_width = input_shape[0], input_shape[1], input_shape[2]
        to_seq_len = from_seq_len
        attention_shape = [input_shape[0], input_shape[1], input_shape[1]]

        # [B, F, N*H] -> [B, N, F, H]
        def transpose_for_scores(input_tensor, seq_len):
            output_shape = [batch_size, seq_len,
                            self.params.num_heads, self.params.size_per_head]
            output_tensor = K.reshape(input_tensor, output_shape)
            return tf.transpose(a=output_tensor, perm=[0, 2, 1, 3])  # [B,N,F,H]

        if self.params.score == True:
            query = self.query_layer(from_tensor)  # [B,F, N*H] [batch_size, from_seq_len, N*H]
            key = self.key_layer(to_tensor)  # [B,T, N*H]
            value = self.key_layer(to_tensor)  # [B,T, N*H]

            query = transpose_for_scores(query, from_seq_len)  # [B, N, F, H]
            key = transpose_for_scores(key, to_seq_len)  # [B, N, T, H]

            attention_scores = tf.matmul(query, key, transpose_b=True)  # [B, N, F, T]
            attention_scores = attention_scores / tf.sqrt(float(self.params.size_per_head))
            return [value, attention_scores]  # [B, F, N*H]
        else:
            value = self.key_layer(to_tensor)  # [B,T, N*H]
            return value
    # noinspection PyUnusedLocal
    def compute_mask(self, inputs, mask=None):
        return mask   # [B, F]

class ProjectionLayer(Layer):
    class Params(Layer.Params):
        hidden_size        = None
        hidden_dropout     = 0.1
        initializer_range  = 0.02
        adapter_size       = None       # bottleneck size of the adapter - arXiv:1902.00751
        adapter_activation = "gelu"
        adapter_init_scale = 1e-3

    def _construct(self, **kwargs):
        super()._construct(**kwargs)
        self.dense      = None
        self.dropout    = None
        self.layer_norm = None

        self.adapter_down = None
        self.adapter_up   = None

        self.supports_masking = True

    # noinspection PyAttributeOutsideInit
    def build(self, input_shape):
        assert isinstance(input_shape, list) and 2 == len(input_shape)
        out_shape, residual_shape = input_shape
        self.input_spec = [keras.layers.InputSpec(shape=out_shape),
                           keras.layers.InputSpec(shape=residual_shape)]

        self.dense = keras.layers.Dense(units=self.params.hidden_size,
                                        kernel_initializer=self.create_initializer(),
                                        name="dense")
        self.dropout    = keras.layers.Dropout(rate=self.params.hidden_dropout)
        self.layer_norm = LayerNormalization(name="LayerNorm")

        if self.params.adapter_size is not None:
            self.adapter_down = keras.layers.Dense(units=self.params.adapter_size,
                                                   kernel_initializer=tf.keras.initializers.TruncatedNormal(
                                                       stddev=self.params.adapter_init_scale),
                                                   activation=self.get_activation(self.params.adapter_activation),
                                                   name="adapter-down")
            self.adapter_up   = keras.layers.Dense(units=self.params.hidden_size,
                                                   kernel_initializer=tf.keras.initializers.TruncatedNormal(
                                                       stddev=self.params.adapter_init_scale),
                                                   name="adapter-up")

        super(ProjectionLayer, self).build(input_shape)

    def call(self, inputs, mask=None, training=None, **kwargs):
        output, residual = inputs
        output = self.dense(output)
        output = self.dropout(output, training=training)

        if self.adapter_down is not None:
            adapted = self.adapter_down(output)
            adapted = self.adapter_up(adapted)
            output = tf.add(output, adapted)

        output = self.layer_norm(tf.add(output, residual))
        return output



class GlobalPool(keras.layers.Layer):
    def __init__(self, **kwargs):

        super().__init__(**kwargs)
        self.supports_masking = True
        self.pooling_op = None
        self.batch_pooling_op = None

    def build(self, input_shape):
        if isinstance(input_shape, list) and len(input_shape) == 2:
            self.data_mode = "disjoint"
        else:
            if len(input_shape) == 2:
                self.data_mode = "single"
            else:
                self.data_mode = "batch"
        super().build(input_shape)

    def call(self, inputs):
        if self.data_mode == "disjoint":
            X = inputs[0]
            I = inputs[1]
            if K.ndim(I) == 2:
                I = I[:, 0]
        else:
            X = inputs

        if self.data_mode == "disjoint":
            return self.pooling_op(X, I)
        else:
            return self.batch_pooling_op(
                X, axis=-2, keepdims=(self.data_mode == "single")
            )

    def compute_output_shape(self, input_shape):
        if self.data_mode == "single":
            return (1,) + input_shape[-1:]
        elif self.data_mode == "batch":
            return input_shape[:-2] + input_shape[-1:]
        else:
            # Input shape is a list of shapes for X and I
            return input_shape[0]
        
class GlobalAvgPool(GlobalPool):
    """
    An average pooling layer. Pools a graph by computing the average of its node
    features.

    **Mode**: single, disjoint, mixed, batch.

    **Input**

    - Node features of shape `([batch], n_nodes, n_node_features)`;
    - Graph IDs of shape `(n_nodes, )` (only in disjoint mode);

    **Output**

    - Pooled node features of shape `(batch, n_node_features)` (if single mode, shape will
    be `(1, n_node_features)`).

    **Arguments**

    None.

    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.pooling_op = tf.math.segment_mean
        self.batch_pooling_op = tf.reduce_mean
        


class Graph_partition(keras.layers.Layer):

    def __init__(self, num_slice=None, size_per_slice=None, **kwargs):
        self.num_slice = num_slice
        self.size_per_slice = size_per_slice
        super(Graph_partition, self).__init__(**kwargs)
    
    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'num_slice': self.num_slice,
            'size_per_slice': self.size_per_slice 
        })
        return config
        
    def call(self, graph ,training=None,**kwargs):
        sliced_graph = []
        for i in range(self.num_slice):
            sliced_graph.append(graph[:,:,i*self.size_per_slice:(i+1)*self.size_per_slice])
        return sliced_graph


    def compute_output_shape(self, input_shape):
        from_shape = input_shape
        output_shape = [self.num_slice, from_shape[0], from_shape[1], self.size_per_slice]
        return output_shape  # [B, F, N*H], [B, F, T]
    
    
def exp_dim(inputs):
    x,exp_shape = inputs
    exp_x =  K.reshape(K.repeat_elements(K.expand_dims(x, axis=1), exp_shape[1] * exp_shape[2], axis=1),
                                 shape=[exp_shape[1],exp_shape[2]])
    return exp_x
    
class STGFU(keras.layers.Layer):

    def __init__(self, **kwargs):

        super(STGFU, self).__init__(**kwargs)
    
    def build(self, input_shape):
        assert isinstance(input_shape, list)
        self.a = self.add_weight(name='a',
                                      shape=(2,),
                                      initializer=keras.initializers.TruncatedNormal(mean=0.0, stddev=0.05, seed=None),
                                      trainable=True)
        self.exp_dim = keras.layers.Lambda(exp_dim)
        self.batch_norm = keras.layers.BatchNormalization()
        self.multiply = keras.layers.Multiply()
        self.add = keras.layers.Add()
        super(STGFU, self).build(input_shape) 
        
    def call(self, inputs ,training=None,**kwargs):
        temporal_graph, spatial_graph = inputs
        
        a = K.softmax(self.a)
        temporal_graph_weight = temporal_graph * a[0]
        spatial_graph_weight = spatial_graph * a[1]
        fusion_out = self.add([temporal_graph_weight,spatial_graph_weight])
        fusion_out = self.batch_norm(fusion_out)
    
        return fusion_out


    def compute_output_shape(self, input_shape):
        return input_shape  # [B, F, N*H], [B, F, T]
    

def slice2parts(x, h1, h2, h3):
 """ Define a tensor slice function
 """
 return x[:,h1:h2, :], x[:,h2:h3,:]


def slice(x, h1, h2):
 """ Define a tensor slice function
 """
 return x[:,h1:h2, :]