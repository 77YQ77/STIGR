
from random import sample
from xmlrpc.client import ExpatParser
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score
import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K
from sklearn.model_selection import StratifiedKFold
from tensorflow.core.protobuf.config_pb2 import OptimizerOptions
from tensorflow import keras
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam, Adadelta, Nadam, RMSprop, Adagrad
from params_flow.activations import gelu
from sklearn.metrics import accuracy_score, confusion_matrix
from keras_gcn.layers import GraphConv
from utils import *

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

    def call(self, graph, training=None, **kwargs):
        sliced_graph = []
        for i in range(self.num_slice):
            sliced_graph.append(graph[:, :, i * self.size_per_slice:(i + 1) * self.size_per_slice])
        return sliced_graph

    def compute_output_shape(self, input_shape):
        from_shape = input_shape
        output_shape = [self.num_slice, from_shape[0], from_shape[1], self.size_per_slice]
        return output_shape  # [B, F, N*H], [B, F, T]


def exp_dim(inputs):
    x, exp_shape = inputs
    exp_x = K.reshape(K.repeat_elements(K.expand_dims(x, axis=1), exp_shape[1] * exp_shape[2], axis=1),
                      shape=[exp_shape[1], exp_shape[2]])
    return exp_x





class DotsimilarityLayer():

    def __call__(self, x1, x2):
        def _dot(x):
            dot1 = K.batch_dot(x[0], x[1], axes=1)
            dot1 = (dot1 * tf.exp(0.07))/x[0].shape[1]
            return dot1

        output_shape = (1,)
        x2 = keras.layers.Lambda(lambda x: K.stop_gradient(x))(x2)
        value = keras.layers.Lambda(_dot, output_shape=output_shape)([x1, x2])
        return value

class cross_fusion(keras.layers.Layer):

    def __init__(self, **kwargs):
        super(cross_fusion, self).__init__(**kwargs)

    def build(self, input_shape):
        assert isinstance(input_shape, list)
        self.a = self.add_weight(name='a',
                                 shape=(2,),
                                 initializer=keras.initializers.TruncatedNormal(mean=0.0, stddev=0.05, seed=None),
                                 trainable=True)
        self.b = self.add_weight(name='b',
                                 shape=(2,),
                                 initializer=keras.initializers.TruncatedNormal(mean=0.0, stddev=0.05, seed=None),
                                 trainable=True)
        self.exp_dim = keras.layers.Lambda(exp_dim)
        self.multiply = keras.layers.Multiply()
        self.add = keras.layers.Add()
        self.batch_norm = keras.layers.BatchNormalization()
        super(cross_fusion, self).build(input_shape)

    def call(self, inputs, training=None, **kwargs):
        temporal_graph, spatial_graph = inputs

        a = K.sigmoid(self.a)
        b = K.sigmoid(self.b)

        temporal_graph_weight = temporal_graph * a[0]
        spatial_graph_weight = spatial_graph * a[1]
        temporal_out = self.add([temporal_graph_weight, spatial_graph_weight])
        temporal_out = self.batch_norm(temporal_out)

        spatial_graph_weight = spatial_graph * b[0]
        temporal_graph4t = temporal_graph * b[1]
        spatial_out = self.add([spatial_graph_weight, temporal_graph4t])
        spatial_out = self.batch_norm(spatial_out)

        return [temporal_out, spatial_out]

    def compute_output_shape(self, input_shape):
        return input_shape  # [B, F, N*H], [B, F, T]


class final_cross_fusion(keras.layers.Layer):

    def __init__(self, **kwargs):
        super(final_cross_fusion, self).__init__(**kwargs)

    def build(self, input_shape):
        assert isinstance(input_shape, list)
        self.a = self.add_weight(name='a',
                                 shape=(2,),
                                 initializer=keras.initializers.TruncatedNormal(mean=0.0, stddev=0.05, seed=None),
                                 trainable=True)
        # self.temporal_fuse = keras.layers.Dense(units=90,
        #                                       kernel_initializer=keras.initializers.TruncatedNormal(mean=0.0, stddev=0.05, seed=None),
        #                                       name="temporal")
        # self.spatial_fuse = keras.layers.Dense(units=90,
        #                                       kernel_initializer=keras.initializers.TruncatedNormal(mean=0.0, stddev=0.05, seed=None),
        #                                       name="spatial")
        self.exp_dim = keras.layers.Lambda(exp_dim)
        self.l2_norm_t = keras.layers.Lambda(lambda x: K.l2_normalize(x, axis=2))
        self.l2_norm_s = keras.layers.Lambda(lambda x: K.l2_normalize(x, axis=1))
        self.batch_norm = keras.layers.BatchNormalization()
        self.add = keras.layers.Add()
        super(final_cross_fusion, self).build(input_shape)

    def call(self, inputs, training=None, **kwargs):
        temporal_graph, spatial_graph = inputs
        # temporal_weight = self.temporal_fuse(temporal_graph)
        # spatial_weight = self.spatial_fuse(spatial_graph)

        # temporal_out = tf.multiply(temporal_graph, temporal_weight)
        # temporal_out = self.batch_norm(temporal_out)

        # spatial_out = tf.multiply(spatial_graph, spatial_weight)
        # spatial_out = self.batch_norm(spatial_out)

        exp_shape = spatial_graph.shape

        a = K.softmax(self.a)
        a_s_exp = self.exp_dim([K.reshape(a[0], (1,)), exp_shape])
        a_exp = self.exp_dim([K.reshape(a[1], (1,)), exp_shape])

        # b = K.softmax(self.b)

        # b_t_exp = self.exp_dim([K.reshape(b[0],(1,)),exp_shape])
        # b_exp = self.exp_dim([K.reshape(b[1],(1,)),exp_shape])

        temporal_graph_weight = temporal_graph * a_exp
        spatial_graph_weight = spatial_graph * a_s_exp
        fused_out = self.add([temporal_graph_weight, spatial_graph_weight])
        # fused_out = self.l2_norm(fused_out)

        # spatial_graph_weight = spatial_graph*b_exp
        # temporal_graph4t = temporal_graph*b_t_exp
        # spatial_out = self.add（)

        return fused_out

    def compute_output_shape(self, input_shape):
        from_shape, b_shape = input_shape
        return from_shape  # [B, F, N*H], [B, F, T]

def my_loss(y_true, y_pred):
    on_diag = tf.linalg.diag_part(y_pred) + (-1)
    on_diag = tf.reduce_sum(tf.pow(on_diag, 2), axis=-1)

    upper_tri = tf.pow(tf.reshape(tf.linalg.band_part(y_pred, 0, -1),[-1]),2)
    lower_tri = tf.pow(tf.reshape(tf.linalg.band_part(y_pred, -1, 0),[-1]),2)
    off_diag = tf.reduce_sum(upper_tri + lower_tri , axis=-1)
    bt_loos = (on_diag + (0.05 * off_diag))/(190+1800)
    # bt_loos = (on_diag + off_diag) / (190*190)
    return bt_loos



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

    def call(self, inputs, training=None, **kwargs):
        temporal_graph, spatial_graph = inputs

        a = K.softmax(self.a)
        temporal_graph_weight = temporal_graph * a[0]
        spatial_graph_weight = spatial_graph * a[1]
        fusion_out = self.add([temporal_graph_weight, spatial_graph_weight])
        fusion_out = self.batch_norm(fusion_out)

        return fusion_out

    def compute_output_shape(self, input_shape):
        return input_shape  # [B, F, N*H], [B, F, T]



def STIGR_ADHD(feature_dim, node_number):
    data_input = keras.layers.Input(shape=(node_number, feature_dim), dtype='float32', name="data_input")
    A_G = keras.layers.Input(shape=(node_number, node_number), dtype='float32', name="AG_input")
    A_P = keras.layers.Input(shape=(node_number, node_number), dtype='float32', name="AP_input")
    A_T = keras.layers.Input(shape=(node_number, node_number), dtype='float32', name="AT_input")
    pheno_input = keras.layers.Input((4,), name='pheno_input')

    ## adj extract

    temporal_input = keras.layers.Permute((2, 1), name='temporal_input')(data_input)

    time_position_embedding = BertEmbeddingsLayer(max_position_embeddings=512, hidden_size=node_number,
                                                  vocab_size=1)(temporal_input)
    spatial_position_embedding = BertEmbeddingsLayer(max_position_embeddings=512, hidden_size=feature_dim,
                                                  vocab_size=1)(data_input)

    time_value_l1, time_attention_l1 = Attention_SV(num_heads=10, size_per_head=19)(time_position_embedding)
    time_attention_prob_l1 = keras.layers.Softmax()(time_attention_l1)
    time_attention_prob_l1 = keras.layers.Dropout(0.2)(time_attention_prob_l1)
    time_attention_score_l1 = MyAttention(num_heads=10, size_per_head=19)([time_value_l1, time_attention_prob_l1])
    projection_time_l1 = ProjectionLayer(hidden_size=node_number)([time_attention_score_l1, time_position_embedding])
    projection_time_l1 = keras.layers.Permute((2,1))(projection_time_l1)


    spatial_value_l1, spatial_attention_l1 = Attention_SV(num_heads=10, size_per_head=9)(spatial_position_embedding)
    spatial_attention_prob_l1 = keras.layers.Softmax()(spatial_attention_l1)
    spatial_attention_prob_l1 = keras.layers.Dropout(0.2)(spatial_attention_prob_l1)
    spatial_attention_score_l1 = MyAttention(num_heads=10, size_per_head=9)(
        [spatial_value_l1, spatial_attention_prob_l1])
    projection_spatial_l1 = ProjectionLayer(hidden_size=feature_dim)(
        [spatial_attention_score_l1, spatial_position_embedding])


    spatial_l1 = keras.layers.Add()([projection_time_l1,projection_spatial_l1])
    temporal_l1 = keras.layers.Permute((2,1))(spatial_l1)


    temporal_intermediate_l1 = keras.layers.Dense(units=4 * node_number, activation=gelu,
                                         kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.02))(
        temporal_l1)
    temporal_out_l1 = ProjectionLayer(hidden_size=node_number)([temporal_intermediate_l1, temporal_l1])

    spatial_intermediate_l1 = keras.layers.Dense(units=4 * feature_dim, activation=gelu,
                                         kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.02))(
        spatial_l1)
    spatial_out_l1 = ProjectionLayer(hidden_size=feature_dim)([spatial_intermediate_l1, spatial_l1])

    time_value_l2, time_attention_l2 = Attention_SV(num_heads=10, size_per_head=19)(temporal_out_l1)
    time_attention_prob_l2 = keras.layers.Softmax()(time_attention_l2)
    time_attention_prob_l2 = keras.layers.Dropout(0.2)(time_attention_prob_l2)
    time_attention_score_l2 = MyAttention(num_heads=10, size_per_head=19)([time_value_l2, time_attention_prob_l2])
    projection_time_l2 = ProjectionLayer(hidden_size=node_number)([time_attention_score_l2, temporal_out_l1])
    projection_time_l2 = keras.layers.Permute((2,1))(projection_time_l2)

    spatial_value_l2, spatial_attention_l2 = Attention_SV(num_heads=10, size_per_head=9)(spatial_out_l1)
    spatial_attention_prob_l2 = keras.layers.Softmax()(spatial_attention_l2)
    spatial_attention_prob_l2 = keras.layers.Dropout(0.2)(spatial_attention_prob_l2)
    spatial_attention_score_l2 = MyAttention(num_heads=10, size_per_head=9)(
        [spatial_value_l2, spatial_attention_prob_l2])
    projection_spatial_l2 = ProjectionLayer(hidden_size=feature_dim)([spatial_attention_score_l2, spatial_out_l1])


    spatial_l2 = keras.layers.Add()([projection_time_l2,projection_spatial_l2])
    temporal_l2 = keras.layers.Permute((2,1))(spatial_l2)


    temporal_intermediate_l2 = keras.layers.Dense(units=4 * node_number, activation=gelu,
                                                  kernel_initializer=tf.keras.initializers.TruncatedNormal(
                                                      stddev=0.02))(temporal_l2)
    temporal_out_l2 = ProjectionLayer(hidden_size=node_number)([temporal_intermediate_l2, temporal_l2])
    temporal_out_l2 = keras.layers.Permute((2, 1))(temporal_out_l2)

    spatial_intermediate_l2 = keras.layers.Dense(units=4 * feature_dim, activation=gelu,
                                                 kernel_initializer=tf.keras.initializers.TruncatedNormal(
                                                     stddev=0.02))(spatial_l2)
    spatial_out_l2 = ProjectionLayer(hidden_size=feature_dim)([spatial_intermediate_l2, spatial_l2])


    adj_sliced_l1 = Graph_partition(6, 15)(spatial_out_l2)
    adj_norm_l1 = []
    adj_dense_norn_l1 = []
    for i, element in enumerate(adj_sliced_l1):
        # dense_norm = keras.layers.LSTM(15,return_sequences=True)(element)
        dense_norm = keras.layers.Dense(15)(element)
        dense_norm = keras.layers.Permute((2, 1))(dense_norm)
        dense_norm = keras.layers.LayerNormalization(axis=-1)(dense_norm)
        adj_dense_norn_l1.append(dense_norm)
        norm = keras.layers.Permute((2, 1))(element)
        norm = keras.layers.LayerNormalization(axis=-1)(norm)
        adj_norm_l1.append(norm)

    adj_sliced_l2 = Graph_partition(3, 30)(spatial_out_l2)
    adj_norm_l2 = []
    adj_dense_norn_l2 = []
    for i, element in enumerate(adj_sliced_l2):
        # dense_norm = keras.layers.LSTM(30,return_sequences=True)(element)
        dense_norm = keras.layers.Dense(30)(element)
        dense_norm = keras.layers.Permute((2, 1))(dense_norm)
        dense_norm = keras.layers.LayerNormalization(axis=-1)(dense_norm)
        adj_dense_norn_l2.append(dense_norm)
        norm = keras.layers.Permute((2, 1))(element)
        norm = keras.layers.LayerNormalization(axis=-1)(norm)
        adj_norm_l2.append(norm)


    # layer 1
    sliced_input1 = Graph_partition(6, 15)(data_input)

    # spatial level
    conv_out_l1 = []
    for i, element in enumerate(sliced_input1):
        if i == 0:
            t1 = K.repeat_elements(sliced_input1[i], 2, axis=1)
            true_input = keras.layers.Concatenate(axis=1)([t1, sliced_input1[i + 1]])
            A_TG2 = DotsimilarityLayer()(adj_dense_norn_l1[i], adj_norm_l1[i + 1])
            A_TG3 = DotsimilarityLayer()(adj_dense_norn_l1[i + 1], adj_norm_l1[i])

            ajd_out1 = keras.layers.Add()([A_T, A_T])
            ajd_out1 = keras.layers.Add()([ajd_out1, A_TG2])
            ajd_out1 = keras.layers.Add()([ajd_out1, A_TG3])

            A1 = keras.layers.Concatenate(axis=2)([A_G, A_T, A_P])
            A2 = keras.layers.Concatenate(axis=2)([A_T, A_G, A_TG2])
            A3 = keras.layers.Concatenate(axis=2)([A_P, A_TG3, A_G])
            edge_input = keras.layers.Concatenate(axis=1)([A1, A2, A3])

            conv_out = GraphConv(
                units=15,
                step_num=1,
            )([true_input, edge_input])
            conv_out = conv_out[:, 190:380, :]
            conv_out_l1.append(conv_out)
        elif i == 5:
            t3 = K.repeat_elements(sliced_input1[i], 2, axis=1)
            true_input = keras.layers.Concatenate(axis=1)([sliced_input1[i - 1], t3])

            A_TG0 = DotsimilarityLayer()(adj_dense_norn_l1[i - 1], adj_norm_l1[i])
            A_TG1 = DotsimilarityLayer()(adj_dense_norn_l1[i], adj_norm_l1[i - 1])

            ajd_out1 = keras.layers.Add()([ajd_out1, A_TG0])
            ajd_out1 = keras.layers.Add()([ajd_out1, A_TG1])
            ajd_out1 = keras.layers.Add()([ajd_out1, A_T])
            ajd_out1 = keras.layers.Add()([ajd_out1, A_T])

            A1 = keras.layers.Concatenate(axis=2)([A_G, A_TG0, A_P])
            A2 = keras.layers.Concatenate(axis=2)([A_TG1, A_G, A_T])
            A3 = keras.layers.Concatenate(axis=2)([A_P, A_T, A_G])
            edge_input = keras.layers.Concatenate(axis=1)([A1, A2, A3])

            conv_out = GraphConv(
                units=15,
                step_num=1,
            )([true_input, edge_input])
            conv_out = conv_out[:, 190:380, :]
            conv_out_l1.append(conv_out)
        else:
            true_input = keras.layers.Concatenate(axis=1)(
                [sliced_input1[i - 1], sliced_input1[i], sliced_input1[i + 1]])

            A_TG0 = DotsimilarityLayer()(adj_dense_norn_l1[i - 1], adj_norm_l1[i])
            A_TG1 = DotsimilarityLayer()(adj_dense_norn_l1[i], adj_norm_l1[i - 1])
            A_TG2 = DotsimilarityLayer()(adj_dense_norn_l1[i], adj_norm_l1[i + 1])
            A_TG3 = DotsimilarityLayer()(adj_dense_norn_l1[i + 1], adj_norm_l1[i])

            ajd_out1 = keras.layers.Add()([ajd_out1, A_TG0])
            ajd_out1 = keras.layers.Add()([ajd_out1, A_TG1])
            ajd_out1 = keras.layers.Add()([ajd_out1, A_TG2])
            ajd_out1 = keras.layers.Add()([ajd_out1, A_TG3])

            A1 = keras.layers.Concatenate(axis=2)([A_G, A_TG0, A_P])
            A2 = keras.layers.Concatenate(axis=2)([A_TG1, A_G, A_TG2])
            A3 = keras.layers.Concatenate(axis=2)([A_P, A_TG3, A_G])
            edge_input = keras.layers.Concatenate(axis=1)([A1, A2, A3])

            conv_out = GraphConv(
                units=15,
                step_num=1,
            )([true_input, edge_input])
            conv_out = conv_out[:, 190:380, :]
            conv_out_l1.append(conv_out)

    conv_out_l1 = keras.layers.Concatenate()(conv_out_l1)
    conv_layer1 = keras.layers.BatchNormalization()(conv_out_l1)
    conv_layer1 = keras.layers.Activation('relu')(conv_layer1)
    conv_layer1 = keras.layers.Dropout(0.5)(conv_layer1)


    # layer 2
    # spatial level
    sliced_input2 = Graph_partition(3, 30)(conv_layer1)

    conv_out_l2 = []
    for i, element in enumerate(sliced_input2):
        if i == 0:
            t1 = K.repeat_elements(sliced_input2[i], 2, axis=1)
            true_input = keras.layers.Concatenate(axis=1)([t1, sliced_input2[i + 1]])

            A_TG2 = DotsimilarityLayer()(adj_dense_norn_l2[i], adj_norm_l2[i + 1])
            A_TG3 = DotsimilarityLayer()(adj_dense_norn_l2[i + 1], adj_norm_l2[i])

            ajd_out2 = keras.layers.Add()([A_T, A_T])
            ajd_out2 = keras.layers.Add()([ajd_out2, A_TG2])
            ajd_out2 = keras.layers.Add()([ajd_out2, A_TG3])

            A1 = keras.layers.Concatenate(axis=2)([A_G, A_T, A_P])
            A2 = keras.layers.Concatenate(axis=2)([A_T, A_G, A_TG2])
            A3 = keras.layers.Concatenate(axis=2)([A_P, A_TG3, A_G])
            edge_input = keras.layers.Concatenate(axis=1)([A1, A2, A3])

            conv_out = GraphConv(
                units=30,
                step_num=1,
            )([true_input, edge_input])
            conv_out = conv_out[:, 190:380, :]
            conv_out_l2.append(conv_out)
        elif i == 2:
            t3 = K.repeat_elements(sliced_input2[i], 2, axis=1)
            true_input = keras.layers.Concatenate(axis=1)([sliced_input2[i - 1], t3])

            A_TG0 = DotsimilarityLayer()(adj_dense_norn_l2[i - 1], adj_norm_l2[i])
            A_TG1 = DotsimilarityLayer()(adj_dense_norn_l2[i], adj_norm_l2[i - 1])

            ajd_out2 = keras.layers.Add()([ajd_out2, A_TG0])
            ajd_out2 = keras.layers.Add()([ajd_out2, A_TG1])
            ajd_out2 = keras.layers.Add()([ajd_out2, A_T])
            ajd_out2 = keras.layers.Add()([ajd_out2, A_T])

            A1 = keras.layers.Concatenate(axis=2)([A_G, A_TG0, A_P])
            A2 = keras.layers.Concatenate(axis=2)([A_TG1, A_G, A_T])
            A3 = keras.layers.Concatenate(axis=2)([A_P, A_T, A_G])
            edge_input = keras.layers.Concatenate(axis=1)([A1, A2, A3])

            conv_out = GraphConv(
                units=30,
                step_num=1,
            )([true_input, edge_input])
            conv_out = conv_out[:, 190:380, :]
            conv_out_l2.append(conv_out)
        else:
            true_input = keras.layers.Concatenate(axis=1)(
                [sliced_input2[i - 1], sliced_input2[i], sliced_input2[i + 1]])

            A_TG0 = DotsimilarityLayer()(adj_dense_norn_l2[i - 1], adj_norm_l2[i])
            A_TG1 = DotsimilarityLayer()(adj_dense_norn_l2[i], adj_norm_l2[i - 1])
            A_TG2 = DotsimilarityLayer()(adj_dense_norn_l2[i], adj_norm_l2[i + 1])
            A_TG3 = DotsimilarityLayer()(adj_dense_norn_l2[i + 1], adj_norm_l2[i])

            ajd_out2 = keras.layers.Add()([ajd_out2, A_TG0])
            ajd_out2 = keras.layers.Add()([ajd_out2, A_TG1])
            ajd_out2 = keras.layers.Add()([ajd_out2, A_TG2])
            ajd_out2 = keras.layers.Add()([ajd_out2, A_TG3])

            A1 = keras.layers.Concatenate(axis=2)([A_G, A_TG0, A_P])
            A2 = keras.layers.Concatenate(axis=2)([A_TG1, A_G, A_TG2])
            A3 = keras.layers.Concatenate(axis=2)([A_P, A_TG3, A_G])
            edge_input = keras.layers.Concatenate(axis=1)([A1, A2, A3])

            conv_out = GraphConv(
                units=30,
                step_num=1,
            )([true_input, edge_input])
            conv_out = conv_out[:, 190:380, :]
            conv_out_l2.append(conv_out)

    conv_out_l2 = keras.layers.Concatenate()(conv_out_l2)
    conv_layer2 = keras.layers.BatchNormalization()(conv_out_l2)
    conv_layer2 = keras.layers.Activation('relu')(conv_layer2)
    conv_layer2 = keras.layers.Dropout(0.5)(conv_layer2)

    fusion_out = keras.layers.Add()([conv_layer2, temporal_out_l2])
    # fusion_out = keras.layers.Permute((2, 1))(fusion_out)

    # global_avg_pooling = keras.layers.GlobalAveragePooling1D()(fusion_out)
    global_avg_pooling = GlobalAvgPool()(fusion_out)

    merge = keras.layers.Concatenate()([global_avg_pooling, pheno_input])
    dense = keras.layers.Dense(64, activation='relu')(merge)
    dense = keras.layers.Dropout(0.5)(dense)
    dense = keras.layers.Dense(10, activation='relu')(dense)
    output = keras.layers.Dense(1, activation='sigmoid',name='output1')(dense)
    adj_final_out = keras.layers.Add(name='adj_output')([ajd_out1 * 0.02, ajd_out2 * 0.04])

    model_ADHD = keras.Model(inputs=[data_input, A_G, A_T, A_P, pheno_input], outputs=[output, adj_final_out])
    # opt = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999)
    # opt = Nadam(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=None, schedule_decay=0.004)
    opt = Adadelta(lr=1.0, rho=0.95, epsilon=None, decay=0.0)
    model_ADHD.compile(optimizer=opt, loss={'output1': 'binary_crossentropy', 'adj_output': my_loss},
                       loss_weights={'output1': 0.5, 'adj_output': 0.5}, metrics=['accuracy'])

    return model_ADHD



def STIGR_ASD(feature_dim, node_number):
    data_input = keras.layers.Input(shape=(node_number, feature_dim), dtype='float32', name="data_input")
    A_G = keras.layers.Input(shape=(node_number, node_number), dtype='float32', name="AG_input")
    A_P = keras.layers.Input(shape=(node_number, node_number), dtype='float32', name="AP_input")
    A_T = keras.layers.Input(shape=(node_number, node_number), dtype='float32', name="AT_input")
    pheno_input = keras.layers.Input((4,), name='pheno_input')

    ## adj extract

    temporal_input = keras.layers.Permute((2, 1), name='temporal_input')(data_input)

    time_position_embedding = BertEmbeddingsLayer(max_position_embeddings=512, hidden_size=node_number,
                                                  vocab_size=1)(temporal_input)
    spatial_position_embedding = BertEmbeddingsLayer(max_position_embeddings=512, hidden_size=feature_dim,
                                                  vocab_size=1)(data_input)

    time_value_l1, time_attention_l1 = Attention_SV(num_heads=10, size_per_head=19)(time_position_embedding)
    time_attention_prob_l1 = keras.layers.Softmax()(time_attention_l1)
    time_attention_prob_l1 = keras.layers.Dropout(0.2)(time_attention_prob_l1)
    time_attention_score_l1 = MyAttention(num_heads=10, size_per_head=19)([time_value_l1, time_attention_prob_l1])
    projection_time_l1 = ProjectionLayer(hidden_size=node_number)([time_attention_score_l1, time_position_embedding])
    projection_time_l1 = keras.layers.Permute((2,1))(projection_time_l1)


    spatial_value_l1, spatial_attention_l1 = Attention_SV(num_heads=10, size_per_head=9)(spatial_position_embedding)
    spatial_attention_prob_l1 = keras.layers.Softmax()(spatial_attention_l1)
    spatial_attention_prob_l1 = keras.layers.Dropout(0.2)(spatial_attention_prob_l1)
    spatial_attention_score_l1 = MyAttention(num_heads=10, size_per_head=9)(
        [spatial_value_l1, spatial_attention_prob_l1])
    projection_spatial_l1 = ProjectionLayer(hidden_size=feature_dim)(
        [spatial_attention_score_l1, spatial_position_embedding])


    spatial_l1 = keras.layers.Add()([projection_time_l1,projection_spatial_l1])
    temporal_l1 = keras.layers.Permute((2,1))(spatial_l1)


    temporal_intermediate_l1 = keras.layers.Dense(units=4 * node_number, activation=gelu,
                                         kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.02))(
        temporal_l1)
    temporal_out_l1 = ProjectionLayer(hidden_size=node_number)([temporal_intermediate_l1, temporal_l1])

    spatial_intermediate_l1 = keras.layers.Dense(units=4 * feature_dim, activation=gelu,
                                         kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.02))(
        spatial_l1)
    spatial_out_l1 = ProjectionLayer(hidden_size=feature_dim)([spatial_intermediate_l1, spatial_l1])

    time_value_l2, time_attention_l2 = Attention_SV(num_heads=10, size_per_head=19)(temporal_out_l1)
    time_attention_prob_l2 = keras.layers.Softmax()(time_attention_l2)
    time_attention_prob_l2 = keras.layers.Dropout(0.2)(time_attention_prob_l2)
    time_attention_score_l2 = MyAttention(num_heads=10, size_per_head=19)([time_value_l2, time_attention_prob_l2])
    projection_time_l2 = ProjectionLayer(hidden_size=node_number)([time_attention_score_l2, temporal_out_l1])
    projection_time_l2 = keras.layers.Permute((2,1))(projection_time_l2)

    spatial_value_l2, spatial_attention_l2 = Attention_SV(num_heads=10, size_per_head=9)(spatial_out_l1)
    spatial_attention_prob_l2 = keras.layers.Softmax()(spatial_attention_l2)
    spatial_attention_prob_l2 = keras.layers.Dropout(0.2)(spatial_attention_prob_l2)
    spatial_attention_score_l2 = MyAttention(num_heads=10, size_per_head=9)(
        [spatial_value_l2, spatial_attention_prob_l2])
    projection_spatial_l2 = ProjectionLayer(hidden_size=feature_dim)([spatial_attention_score_l2, spatial_out_l1])


    spatial_l2 = keras.layers.Add()([projection_time_l2,projection_spatial_l2])
    temporal_l2 = keras.layers.Permute((2,1))(spatial_l2)


    temporal_intermediate_l2 = keras.layers.Dense(units=4 * node_number, activation=gelu,
                                                  kernel_initializer=tf.keras.initializers.TruncatedNormal(
                                                      stddev=0.02))(temporal_l2)
    temporal_out_l2 = ProjectionLayer(hidden_size=node_number)([temporal_intermediate_l2, temporal_l2])
    temporal_out_l2 = keras.layers.Permute((2, 1))(temporal_out_l2)

    spatial_intermediate_l2 = keras.layers.Dense(units=4 * feature_dim, activation=gelu,
                                                 kernel_initializer=tf.keras.initializers.TruncatedNormal(
                                                     stddev=0.02))(spatial_l2)
    spatial_out_l2 = ProjectionLayer(hidden_size=feature_dim)([spatial_intermediate_l2, spatial_l2])


    adj_sliced_l1 = Graph_partition(6, 15)(spatial_out_l2)
    adj_norm_l1 = []
    adj_dense_norn_l1 = []
    for i, element in enumerate(adj_sliced_l1):
        # dense_norm = keras.layers.LSTM(15,return_sequences=True)(element)
        dense_norm = keras.layers.Dense(15)(element)
        dense_norm = keras.layers.Permute((2, 1))(dense_norm)
        dense_norm = keras.layers.LayerNormalization(axis=-1)(dense_norm)
        adj_dense_norn_l1.append(dense_norm)
        norm = keras.layers.Permute((2, 1))(element)
        norm = keras.layers.LayerNormalization(axis=-1)(norm)
        adj_norm_l1.append(norm)

    adj_sliced_l2 = Graph_partition(3, 30)(spatial_out_l2)
    adj_norm_l2 = []
    adj_dense_norn_l2 = []
    for i, element in enumerate(adj_sliced_l2):
        # dense_norm = keras.layers.LSTM(30,return_sequences=True)(element)
        dense_norm = keras.layers.Dense(30)(element)
        dense_norm = keras.layers.Permute((2, 1))(dense_norm)
        dense_norm = keras.layers.LayerNormalization(axis=-1)(dense_norm)
        adj_dense_norn_l2.append(dense_norm)
        norm = keras.layers.Permute((2, 1))(element)
        norm = keras.layers.LayerNormalization(axis=-1)(norm)
        adj_norm_l2.append(norm)


    # layer 1
    sliced_input1 = Graph_partition(6, 15)(data_input)

    # spatial level
    conv_out_l1 = []
    for i, element in enumerate(sliced_input1):
        if i == 0:
            t1 = K.repeat_elements(sliced_input1[i], 2, axis=1)
            true_input = keras.layers.Concatenate(axis=1)([t1, sliced_input1[i + 1]])
            A_TG2 = DotsimilarityLayer()(adj_dense_norn_l1[i], adj_norm_l1[i + 1])
            A_TG3 = DotsimilarityLayer()(adj_dense_norn_l1[i + 1], adj_norm_l1[i])

            ajd_out1 = keras.layers.Add()([A_T, A_T])
            ajd_out1 = keras.layers.Add()([ajd_out1, A_TG2])
            ajd_out1 = keras.layers.Add()([ajd_out1, A_TG3])

            A1 = keras.layers.Concatenate(axis=2)([A_G, A_T, A_P])
            A2 = keras.layers.Concatenate(axis=2)([A_T, A_G, A_TG2])
            A3 = keras.layers.Concatenate(axis=2)([A_P, A_TG3, A_G])
            edge_input = keras.layers.Concatenate(axis=1)([A1, A2, A3])

            conv_out = GraphConv(
                units=15,
                step_num=1,
            )([true_input, edge_input])
            conv_out = conv_out[:, 190:380, :]
            conv_out_l1.append(conv_out)
        elif i == 5:
            t3 = K.repeat_elements(sliced_input1[i], 2, axis=1)
            true_input = keras.layers.Concatenate(axis=1)([sliced_input1[i - 1], t3])

            A_TG0 = DotsimilarityLayer()(adj_dense_norn_l1[i - 1], adj_norm_l1[i])
            A_TG1 = DotsimilarityLayer()(adj_dense_norn_l1[i], adj_norm_l1[i - 1])

            ajd_out1 = keras.layers.Add()([ajd_out1, A_TG0])
            ajd_out1 = keras.layers.Add()([ajd_out1, A_TG1])
            ajd_out1 = keras.layers.Add()([ajd_out1, A_T])
            ajd_out1 = keras.layers.Add()([ajd_out1, A_T])

            A1 = keras.layers.Concatenate(axis=2)([A_G, A_TG0, A_P])
            A2 = keras.layers.Concatenate(axis=2)([A_TG1, A_G, A_T])
            A3 = keras.layers.Concatenate(axis=2)([A_P, A_T, A_G])
            edge_input = keras.layers.Concatenate(axis=1)([A1, A2, A3])

            conv_out = GraphConv(
                units=15,
                step_num=1,
            )([true_input, edge_input])
            conv_out = conv_out[:, 190:380, :]
            conv_out_l1.append(conv_out)
        else:
            true_input = keras.layers.Concatenate(axis=1)(
                [sliced_input1[i - 1], sliced_input1[i], sliced_input1[i + 1]])

            A_TG0 = DotsimilarityLayer()(adj_dense_norn_l1[i - 1], adj_norm_l1[i])
            A_TG1 = DotsimilarityLayer()(adj_dense_norn_l1[i], adj_norm_l1[i - 1])
            A_TG2 = DotsimilarityLayer()(adj_dense_norn_l1[i], adj_norm_l1[i + 1])
            A_TG3 = DotsimilarityLayer()(adj_dense_norn_l1[i + 1], adj_norm_l1[i])

            ajd_out1 = keras.layers.Add()([ajd_out1, A_TG0])
            ajd_out1 = keras.layers.Add()([ajd_out1, A_TG1])
            ajd_out1 = keras.layers.Add()([ajd_out1, A_TG2])
            ajd_out1 = keras.layers.Add()([ajd_out1, A_TG3])

            A1 = keras.layers.Concatenate(axis=2)([A_G, A_TG0, A_P])
            A2 = keras.layers.Concatenate(axis=2)([A_TG1, A_G, A_TG2])
            A3 = keras.layers.Concatenate(axis=2)([A_P, A_TG3, A_G])
            edge_input = keras.layers.Concatenate(axis=1)([A1, A2, A3])

            conv_out = GraphConv(
                units=15,
                step_num=1,
            )([true_input, edge_input])
            conv_out = conv_out[:, 190:380, :]
            conv_out_l1.append(conv_out)

    conv_out_l1 = keras.layers.Concatenate()(conv_out_l1)
    conv_layer1 = keras.layers.BatchNormalization()(conv_out_l1)
    conv_layer1 = keras.layers.Activation('relu')(conv_layer1)
    conv_layer1 = keras.layers.Dropout(0.5)(conv_layer1)


    # layer 2
    # spatial level
    sliced_input2 = Graph_partition(3, 30)(conv_layer1)

    conv_out_l2 = []
    for i, element in enumerate(sliced_input2):
        if i == 0:
            t1 = K.repeat_elements(sliced_input2[i], 2, axis=1)
            true_input = keras.layers.Concatenate(axis=1)([t1, sliced_input2[i + 1]])

            A_TG2 = DotsimilarityLayer()(adj_dense_norn_l2[i], adj_norm_l2[i + 1])
            A_TG3 = DotsimilarityLayer()(adj_dense_norn_l2[i + 1], adj_norm_l2[i])

            ajd_out2 = keras.layers.Add()([A_T, A_T])
            ajd_out2 = keras.layers.Add()([ajd_out2, A_TG2])
            ajd_out2 = keras.layers.Add()([ajd_out2, A_TG3])

            A1 = keras.layers.Concatenate(axis=2)([A_G, A_T, A_P])
            A2 = keras.layers.Concatenate(axis=2)([A_T, A_G, A_TG2])
            A3 = keras.layers.Concatenate(axis=2)([A_P, A_TG3, A_G])
            edge_input = keras.layers.Concatenate(axis=1)([A1, A2, A3])

            conv_out = GraphConv(
                units=30,
                step_num=1,
            )([true_input, edge_input])
            conv_out = conv_out[:, 190:380, :]
            conv_out_l2.append(conv_out)
        elif i == 2:
            t3 = K.repeat_elements(sliced_input2[i], 2, axis=1)
            true_input = keras.layers.Concatenate(axis=1)([sliced_input2[i - 1], t3])

            A_TG0 = DotsimilarityLayer()(adj_dense_norn_l2[i - 1], adj_norm_l2[i])
            A_TG1 = DotsimilarityLayer()(adj_dense_norn_l2[i], adj_norm_l2[i - 1])

            ajd_out2 = keras.layers.Add()([ajd_out2, A_TG0])
            ajd_out2 = keras.layers.Add()([ajd_out2, A_TG1])
            ajd_out2 = keras.layers.Add()([ajd_out2, A_T])
            ajd_out2 = keras.layers.Add()([ajd_out2, A_T])

            A1 = keras.layers.Concatenate(axis=2)([A_G, A_TG0, A_P])
            A2 = keras.layers.Concatenate(axis=2)([A_TG1, A_G, A_T])
            A3 = keras.layers.Concatenate(axis=2)([A_P, A_T, A_G])
            edge_input = keras.layers.Concatenate(axis=1)([A1, A2, A3])

            conv_out = GraphConv(
                units=30,
                step_num=1,
            )([true_input, edge_input])
            conv_out = conv_out[:, 190:380, :]
            conv_out_l2.append(conv_out)
        else:
            true_input = keras.layers.Concatenate(axis=1)(
                [sliced_input2[i - 1], sliced_input2[i], sliced_input2[i + 1]])

            A_TG0 = DotsimilarityLayer()(adj_dense_norn_l2[i - 1], adj_norm_l2[i])
            A_TG1 = DotsimilarityLayer()(adj_dense_norn_l2[i], adj_norm_l2[i - 1])
            A_TG2 = DotsimilarityLayer()(adj_dense_norn_l2[i], adj_norm_l2[i + 1])
            A_TG3 = DotsimilarityLayer()(adj_dense_norn_l2[i + 1], adj_norm_l2[i])

            ajd_out2 = keras.layers.Add()([ajd_out2, A_TG0])
            ajd_out2 = keras.layers.Add()([ajd_out2, A_TG1])
            ajd_out2 = keras.layers.Add()([ajd_out2, A_TG2])
            ajd_out2 = keras.layers.Add()([ajd_out2, A_TG3])

            A1 = keras.layers.Concatenate(axis=2)([A_G, A_TG0, A_P])
            A2 = keras.layers.Concatenate(axis=2)([A_TG1, A_G, A_TG2])
            A3 = keras.layers.Concatenate(axis=2)([A_P, A_TG3, A_G])
            edge_input = keras.layers.Concatenate(axis=1)([A1, A2, A3])

            conv_out = GraphConv(
                units=30,
                step_num=1,
            )([true_input, edge_input])
            conv_out = conv_out[:, 190:380, :]
            conv_out_l2.append(conv_out)

    conv_out_l2 = keras.layers.Concatenate()(conv_out_l2)
    conv_layer2 = keras.layers.BatchNormalization()(conv_out_l2)
    conv_layer2 = keras.layers.Activation('relu')(conv_layer2)
    conv_layer2 = keras.layers.Dropout(0.5)(conv_layer2)

    fusion_out = keras.layers.Add()([conv_layer2, temporal_out_l2])
    fusion_out = keras.layers.Permute((2, 1))(fusion_out)

    # global_avg_pooling = keras.layers.GlobalAveragePooling1D()(fusion_out)
    global_avg_pooling = GlobalAvgPool()(fusion_out)

    merge = keras.layers.Concatenate()([global_avg_pooling, pheno_input])
    dense = keras.layers.Dense(64, activation='relu')(merge)
    dense = keras.layers.Dropout(0.5)(dense)
    dense = keras.layers.Dense(10, activation='relu')(dense)
    output = keras.layers.Dense(1, activation='sigmoid',name='output1')(dense)
    adj_final_out = keras.layers.Add(name='adj_output')([ajd_out1 * 0.02, ajd_out2 * 0.04])

    model_ASD = keras.Model(inputs=[data_input, A_G, A_T, A_P, pheno_input], outputs=[output, adj_final_out])
    # opt = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999)
    # opt = Nadam(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=None, schedule_decay=0.004)
    opt = Adadelta(lr=1.0, rho=0.95, epsilon=None, decay=0.0)
    model_ASD.compile(optimizer=opt, loss={'output1': 'binary_crossentropy', 'adj_output': my_loss},
                       loss_weights={'output1': 0.5, 'adj_output': 0.5}, metrics=['accuracy'])

    return model_ASD

