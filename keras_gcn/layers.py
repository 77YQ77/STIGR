from tensorflow import keras
from tensorflow.keras import backend as K
from tensorflow.keras import activations, constraints, initializers, regularizers
import tensorflow as tf
import keras_gcn.ops as ops


LAYER_KWARGS = {"activation", "use_bias"}
KERAS_KWARGS = {
    "trainable",
    "name",
    "dtype",
    "dynamic",
    "input_dim",
    "input_shape",
    "batch_input_shape",
    "batch_size",
    "weights",
    "activity_regularizer",
    "autocast",
    "implementation",
}


def is_layer_kwarg(key):
    return key not in KERAS_KWARGS and (
        key.endswith("_initializer")
        or key.endswith("_regularizer")
        or key.endswith("_constraint")
        or key in LAYER_KWARGS
    )


def is_keras_kwarg(key):
    return key in KERAS_KWARGS


def deserialize_kwarg(key, attr):
    if key.endswith("_initializer"):
        return initializers.get(attr)
    if key.endswith("_regularizer"):
        return regularizers.get(attr)
    if key.endswith("_constraint"):
        return constraints.get(attr)
    if key == "activation":
        return activations.get(attr)
    return attr


def serialize_kwarg(key, attr):
    if key.endswith("_initializer"):
        return initializers.serialize(attr)
    if key.endswith("_regularizer"):
        return regularizers.serialize(attr)
    if key.endswith("_constraint"):
        return constraints.serialize(attr)
    if key == "activation":
        return activations.serialize(attr)
    if key == "use_bias":
        return attr
    
class Pool(keras.layers.Layer):
    r"""
    A general class for pooling layers.
    You can extend this class to create custom implementations of pooling layers.
    Any extension of this class must implement the `call(self, inputs)` and
    `config(self)` methods.
    **Arguments**:
    - ``**kwargs`: additional keyword arguments specific to Keras' Layers, like
    regularizers, initializers, constraints, etc.
    """

    def __init__(self, **kwargs):
        super().__init__(**{k: v for k, v in kwargs.items() if is_keras_kwarg(k)})
        self.supports_masking = True
        self.kwargs_keys = []
        for key in kwargs:
            if is_layer_kwarg(key):
                attr = kwargs[key]
                attr = deserialize_kwarg(key, attr)
                self.kwargs_keys.append(key)
                setattr(self, key, attr)

    def build(self, input_shape):
        self.built = True

    def call(self, inputs):
        raise NotImplementedError

    def get_config(self):
        base_config = super().get_config()
        keras_config = {}
        for key in self.kwargs_keys:
            keras_config[key] = serialize_kwarg(key, getattr(self, key))
        return {**base_config, **keras_config, **self.config}

    def compute_mask(self, inputs, mask=None):
        if mask is not None:
            return None

    @property
    def config(self):
        return {}

class TopKPool(Pool):
    r"""
    A gPool/Top-K layer from the papers
    > [Graph U-Nets](https://arxiv.org/abs/1905.05178)<br>
    > Hongyang Gao and Shuiwang Ji
    and
    > [Towards Sparse Hierarchical Graph Classifiers](https://arxiv.org/abs/1811.01287)<br>
    > Cătălina Cangea et al.
    **Mode**: single, disjoint.
    This layer computes the following operations:
    $$
        \y = \frac{\X\p}{\|\p\|}; \;\;\;\;
        \i = \textrm{rank}(\y, K); \;\;\;\;
        \X' = (\X \odot \textrm{tanh}(\y))_\i; \;\;\;\;
        \A' = \A_{\i, \i}
    $$
    where \(\textrm{rank}(\y, K)\) returns the indices of the top K values of
    \(\y\), and \(\p\) is a learnable parameter vector of size \(F\).
    \(K\) is defined for each graph as a fraction of the number of nodes,
    controlled by the `ratio` argument.
    Note that the the gating operation \(\textrm{tanh}(\y)\) (Cangea et al.)
    can be replaced with a sigmoid (Gao & Ji).
    **Input**
    - Node features of shape `(n_nodes, n_node_features)`;
    - Binary adjacency matrix of shape `(n_nodes, n_nodes)`;
    - Graph IDs of shape `(n_nodes, )` (only in disjoint mode);
    **Output**
    - Reduced node features of shape `(ratio * n_nodes, n_node_features)`;
    - Reduced adjacency matrix of shape `(ratio * n_nodes, ratio * n_nodes)`;
    - Reduced graph IDs of shape `(ratio * n_nodes, )` (only in disjoint mode);
    - If `return_mask=True`, the binary pooling mask of shape `(ratio * n_nodes, )`.
    **Arguments**
    - `ratio`: float between 0 and 1, ratio of nodes to keep in each graph;
    - `return_mask`: boolean, whether to return the binary mask used for pooling;
    - `sigmoid_gating`: boolean, use a sigmoid gating activation instead of a
        tanh;
    - `kernel_initializer`: initializer for the weights;
    - `kernel_regularizer`: regularization applied to the weights;
    - `kernel_constraint`: constraint applied to the weights;
    """

    def __init__(
        self,
        ratio,
        return_mask=False,
        sigmoid_gating=False,
        kernel_initializer="glorot_uniform",
        kernel_regularizer=None,
        kernel_constraint=None,
        **kwargs
    ):
        super().__init__(
            kernel_initializer=kernel_initializer,
            kernel_regularizer=kernel_regularizer,
            kernel_constraint=kernel_constraint,
            **kwargs
        )
        self.ratio = ratio
        self.return_mask = return_mask
        self.sigmoid_gating = sigmoid_gating
        self.gating_op = K.sigmoid if self.sigmoid_gating else K.tanh

    def build(self, input_shape):
        print(input_shape[0])
        self.F = int(input_shape[0][-1])
        self.N = int(input_shape[0][1])
        self.kernel = self.add_weight(
            shape=(self.F, 1),
            name="kernel",
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
        )
        super().build(input_shape)

    def call(self, inputs):
        if len(inputs) == 3:
            X, A, I = inputs
        else:
            X, A = inputs
            I = tf.zeros(tf.shape(X)[:1])
            
        if K.ndim(I) == 2:
            I = I[:, 0]
        I = tf.cast(I, tf.int32)

        A_is_sparse = K.is_sparse(A)

        # Get mask
        y = self.compute_scores(X, A, I)
        N = K.shape(X)[-2]
        indices = ops.segment_top_k(y[:, 0], I, self.ratio)
        indices = tf.sort(indices)  # required for ordered SparseTensors
        mask = ops.indices_to_mask(indices, N)

        # Multiply X and y to make layer differentiable
        features = X * self.gating_op(y)

        axis = 0 if len(A.shape) == 2 else 1  # Cannot use negative axis
        # Reduce X
        X_pooled = tf.gather(features, indices, axis=axis)

        # Reduce A
        if A_is_sparse:
            A_pooled, _ = ops.gather_sparse_square(A, indices, mask=mask)
        else:
            A_pooled = tf.gather(A, indices, axis=axis)
            A_pooled = tf.gather(A_pooled, indices, axis=axis + 1)

        output = [X_pooled, A_pooled]

        # Reduce I


        if self.return_mask:
            output.append(mask)

        return output

    def compute_scores(self, X, A, I):
        return K.dot(X, K.l2_normalize(self.kernel))

    @property
    def config(self):
        return {
            "ratio": self.ratio,
            "return_mask": self.return_mask,
            "sigmoid_gating": self.sigmoid_gating,
        }
        
class SAGPool(TopKPool):
    r"""
    A self-attention graph pooling layer from the paper
    > [Self-Attention Graph Pooling](https://arxiv.org/abs/1904.08082)<br>
    > Junhyun Lee et al.
    **Mode**: single, disjoint.
    This layer computes the following operations:
    $$
        \y = \textrm{GNN}(\A, \X); \;\;\;\;
        \i = \textrm{rank}(\y, K); \;\;\;\;
        \X' = (\X \odot \textrm{tanh}(\y))_\i; \;\;\;\;
        \A' = \A_{\i, \i}
    $$
    where \(\textrm{rank}(\y, K)\) returns the indices of the top K values of
    \(\y\) and
    $$
        \textrm{GNN}(\A, \X) = \A \X \W.
    $$
    \(K\) is defined for each graph as a fraction of the number of nodes,
    controlled by the `ratio` argument.
    **Input**
    - Node features of shape `(n_nodes, n_node_features)`;
    - Binary adjacency matrix of shape `(n_nodes, n_nodes)`;
    - Graph IDs of shape `(n_nodes, )` (only in disjoint mode);
    **Output**
    - Reduced node features of shape `(ratio * n_nodes, n_node_features)`;
    - Reduced adjacency matrix of shape `(ratio * n_nodes, ratio * n_nodes)`;
    - Reduced graph IDs of shape `(ratio * n_nodes, )` (only in disjoint mode);
    - If `return_mask=True`, the binary pooling mask of shape `(ratio * n_nodes, )`.
    **Arguments**
    - `ratio`: float between 0 and 1, ratio of nodes to keep in each graph;
    - `return_mask`: boolean, whether to return the binary mask used for pooling;
    - `sigmoid_gating`: boolean, use a sigmoid gating activation instead of a
        tanh;
    - `kernel_initializer`: initializer for the weights;
    - `kernel_regularizer`: regularization applied to the weights;
    - `kernel_constraint`: constraint applied to the weights;
    """

    def __init__(
        self,
        ratio,
        return_mask=False,
        sigmoid_gating=False,
        kernel_initializer="glorot_uniform",
        kernel_regularizer=None,
        kernel_constraint=None,
        **kwargs
    ):
        super().__init__(
            ratio,
            return_mask=return_mask,
            sigmoid_gating=sigmoid_gating,
            kernel_initializer=kernel_initializer,
            kernel_regularizer=kernel_regularizer,
            kernel_constraint=kernel_constraint,
            **kwargs
        )

    def compute_scores(self, X, A, I):
        scores = K.dot(X, self.kernel)
        scores = ops.modal_dot(A, scores)
        return scores

class GraphLayer(keras.layers.Layer):

    def __init__(self,
                 step_num=1,
                 activation=None,
                 **kwargs):
        """Initialize the layer.

        :param step_num: Two nodes are considered as connected if they could be reached in `step_num` steps.
        :param activation: The activation function after convolution.
        :param kwargs: Other arguments for parent class.
        """
        self.supports_masking = True
        self.step_num = step_num
        self.activation = keras.activations.get(activation)
        self.supports_masking = True
        super(GraphLayer, self).__init__(**kwargs)

    def get_config(self):
        config = {
            'step_num': self.step_num,
            'activation': self.activation,
        }
        base_config = super(GraphLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def _get_walked_edges(self, edges, step_num):
        """Get the connection graph within `step_num` steps

        :param edges: The graph in single step.
        :param step_num: Number of steps.
        :return: The new graph that has the same shape with `edges`.
        """
        if step_num <= 1:
            return edges
        deeper = self._get_walked_edges(K.batch_dot(edges, edges), step_num // 2)
        if step_num % 2 == 1:
            deeper += edges
        return K.cast(K.greater(deeper, 0.0), K.floatx())

    def call(self, inputs, **kwargs):
        features, edges = inputs
        edges = K.cast(edges, K.floatx())
        if self.step_num > 1:
            edges = self._get_walked_edges(edges, self.step_num)
        outputs = self.activation(self._call(features, edges))
        return outputs

    def _call(self, features, edges):
        raise NotImplementedError('The class is not intended to be used directly.')


class GraphConv(GraphLayer):
    r"""Graph convolutional layer.

    h_i^{(t)} = \sigma \left ( \frac{ G_i^T (h_i^{(t - 1)} W + b)}{\sum G_i}  \right )
    """

    def __init__(self,
                 units,
                 kernel_initializer='glorot_uniform',
                 kernel_regularizer=None,
                 kernel_constraint=None,
                 use_bias=True,
                 bias_initializer='zeros',
                 bias_regularizer=None,
                 bias_constraint=None,
                 **kwargs):
        """Initialize the layer.

        :param units: Number of new states. If the input shape is (batch_size, node_num, feature_len), then the output
                      shape is (batch_size, node_num, units).
        :param kernel_initializer: The initializer of the kernel weight matrix.
        :param kernel_regularizer: The regularizer of the kernel weight matrix.
        :param kernel_constraint:  The constraint of the kernel weight matrix.
        :param use_bias: Whether to use bias term.
        :param bias_initializer: The initializer of the bias vector.
        :param bias_regularizer: The regularizer of the bias vector.
        :param bias_constraint: The constraint of the bias vector.
        :param kwargs: Other arguments for parent class.
        """
        self.units = units
        self.kernel_initializer = keras.initializers.get(kernel_initializer)
        self.kernel_regularizer = keras.regularizers.get(kernel_regularizer)
        self.kernel_constraint = keras.constraints.get(kernel_constraint)
        self.use_bias = use_bias
        self.bias_initializer = keras.initializers.get(bias_initializer)
        self.bias_regularizer = keras.regularizers.get(bias_regularizer)
        self.bias_constraint = keras.constraints.get(bias_constraint)

        self.W, self.b = None, None
        super(GraphConv, self).__init__(**kwargs)

    def get_config(self):
        config = {
            'units': self.units,
            'kernel_initializer': keras.initializers.serialize(self.kernel_initializer),
            'kernel_regularizer': keras.regularizers.serialize(self.kernel_regularizer),
            'kernel_constraint': keras.constraints.serialize(self.kernel_constraint),
            'use_bias': self.use_bias,
            'bias_initializer': keras.initializers.serialize(self.bias_initializer),
            'bias_regularizer': keras.regularizers.serialize(self.bias_regularizer),
            'bias_constraint': keras.constraints.serialize(self.bias_constraint),
        }
        base_config = super(GraphConv, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def build(self, input_shape):
        feature_dim = int(input_shape[0][-1])
        self.W = self.add_weight(
            shape=(feature_dim, self.units),
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
            name='{}_W'.format(self.name),
        )
        if self.use_bias:
            self.b = self.add_weight(
                shape=(self.units,),
                initializer=self.bias_initializer,
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint,
                name='{}_b'.format(self.name),
            )
        super(GraphConv, self).build(input_shape)

    def compute_output_shape(self, input_shape):
        return input_shape[0][:2] + (self.units,)

    def compute_mask(self, inputs, mask=None):
        if mask is None:
            mask = [None]
        return mask[0]

    def _call(self, features, edges):
        features = K.dot(features, self.W)
        if self.use_bias:
            features += self.b
        if self.step_num > 1:
            edges = self._get_walked_edges(edges, self.step_num)
        return K.batch_dot(K.permute_dimensions(edges, (0, 2, 1)), features) \
            / (K.sum(edges, axis=2, keepdims=True) + K.epsilon())


class GraphGIN(GraphLayer):
    r"""Graph convolutional layer.

    h_i^{(t)} = \sigma \left ( \frac{ G_i^T (h_i^{(t - 1)} W + b)}{\sum G_i}  \right )
    """

    def __init__(self,
                 units,
                 input_dim,
                 hidden_dim,
                 kernel_initializer='glorot_uniform',
                 kernel_regularizer=None,
                 kernel_constraint=None,
                 use_bias=True,
                 bias_initializer='zeros',
                 bias_regularizer=None,
                 bias_constraint=None,
                 **kwargs):
        """Initialize the layer.

        :param units: Number of new states. If the input shape is (batch_size, node_num, feature_len), then the output
                      shape is (batch_size, node_num, units).
        :param kernel_initializer: The initializer of the kernel weight matrix.
        :param kernel_regularizer: The regularizer of the kernel weight matrix.
        :param kernel_constraint:  The constraint of the kernel weight matrix.
        :param use_bias: Whether to use bias term.
        :param bias_initializer: The initializer of the bias vector.
        :param bias_regularizer: The regularizer of the bias vector.
        :param bias_constraint: The constraint of the bias vector.
        :param kwargs: Other arguments for parent class.
        """
        self.units = units
        self.kernel_initializer = keras.initializers.get(kernel_initializer)
        self.kernel_regularizer = keras.regularizers.get(kernel_regularizer)
        self.kernel_constraint = keras.constraints.get(kernel_constraint)
        self.use_bias = use_bias
        self.bias_initializer = keras.initializers.get(bias_initializer)
        self.bias_regularizer = keras.regularizers.get(bias_regularizer)
        self.bias_constraint = keras.constraints.get(bias_constraint)
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.W, self.b = None, None
        super(GraphGIN, self).__init__(**kwargs)

    def get_config(self):
        config = {
            'units': self.units,
            'kernel_initializer': keras.initializers.serialize(self.kernel_initializer),
            'kernel_regularizer': keras.regularizers.serialize(self.kernel_regularizer),
            'kernel_constraint': keras.constraints.serialize(self.kernel_constraint),
            'use_bias': self.use_bias,
            'bias_initializer': keras.initializers.serialize(self.bias_initializer),
            'bias_regularizer': keras.regularizers.serialize(self.bias_regularizer),
            'bias_constraint': keras.constraints.serialize(self.bias_constraint),
        }
        base_config = super(GraphGIN, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def build(self, input_shape):
        feature_dim = int(input_shape[0][-1])
        self.W = self.add_weight(
            shape=(feature_dim, self.units),
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
            name='{}_W'.format(self.name),
        )
        if self.use_bias:
            self.b = self.add_weight(
                shape=(self.units,),
                initializer=self.bias_initializer,
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint,
                name='{}_b'.format(self.name),
            )
        self.fc1 = keras.layers.Dense(units=self.hidden_dim)
        self.fc2 = keras.layers.Dense(units=self.input_dim)
        self.batchnorm1 = keras.layers.BatchNormalization()
        self.batchnorm2 = keras.layers.BatchNormalization()
        self.activation1 = keras.layers.Activation('relu')
        self.activation2 = keras.layers.Activation('relu')

        super(GraphGIN, self).build(input_shape)

    def compute_output_shape(self, input_shape):
        return input_shape[0][:2] + (self.units,)

    def compute_mask(self, inputs, mask=None):
        if mask is None:
            mask = [None]
        return mask[0]

    def _call(self, features, edges):
        features = K.dot(features, self.W)
        if self.use_bias:
            features += self.b
        if self.step_num > 1:
            edges = self._get_walked_edges(edges, self.step_num)
        out = K.batch_dot(K.permute_dimensions(edges, (0, 2, 1)), features) \
            / (K.sum(edges, axis=2, keepdims=True) + K.epsilon())
        
        out = self.fc1(out)
        out = self.batchnorm1(out)
        out = self.activation1(out)
        out = self.fc2(out)
        out = self.activation2(out)
        out = self.batchnorm2(out)
        return out

class GraphPool(GraphLayer):

    def compute_output_shape(self, input_shape):
        return input_shape

    def compute_mask(self, inputs, mask=None):
        if mask is None:
            mask = [None]
        return mask[0]


class GraphMaxPool(GraphPool):

    NEG_INF = -1e38

    def _call(self, features, edges):
        node_num = K.shape(features)[1]
        features = K.tile(K.expand_dims(features, axis=1), [1, node_num, 1, 1]) \
            + K.expand_dims((1.0 - edges) * self.NEG_INF, axis=-1)
        return K.max(features, axis=2)


class GraphAveragePool(GraphPool):

    def _call(self, features, edges):
        return K.batch_dot(K.permute_dimensions(edges, (0, 2, 1)), features) \
            / (K.sum(edges, axis=2, keepdims=True) + K.epsilon())
