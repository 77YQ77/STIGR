from random import sample
from xmlrpc.client import ExpatParser
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score
import numpy as np
import tensorflow as tf
import random
from tensorflow.keras import backend as K
from sklearn.model_selection import StratifiedKFold
from tensorflow.core.protobuf.config_pb2 import OptimizerOptions
from tensorflow import keras
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import sklearn.metrics as metrics
from tensorflow.keras.optimizers import Adam, Adadelta, Nadam, RMSprop, Adagrad
import os
from params_flow.activations import gelu
from sklearn.metrics import accuracy_score, confusion_matrix
from keras_gcn.layers import GraphConv
import pandas as pd
import gc
from utils import *
# import models
from STIGR import STIGR_ADHD

print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

os.environ['PYTHONHASHSEED'] = str(42)
random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)

# tf.compat.v1.disable_eager_execution()
# print(tf.__version__)



def data_augment(data, label, pheno, crops):
    augment_label = []
    augment_data = []
    augmented_pheno = []
    sk_data = []
    sk_label = []
    for i in range(len(data)):
        max = data[i].shape[0]
        if max >= 100:
            sk_data.append(data[i])
            sk_label.append(label[i])

            range_list = range(90 + 1, int(max))
            random_index = random.sample(range_list, crops)
            for j in range(crops):
                r = random_index[j]
                # r = random.randint(90,int(max))
                augment_data.append(data[i][r - 90:r])
                augment_label.append(label[i])
                augmented_pheno.append(pheno[i])

    return np.array(augment_data), np.array(augment_label), np.array(augmented_pheno), np.array(sk_data), np.array(
        sk_label)


def Persion_matrix(data):
    FC = np.corrcoef(data, rowvar=0)
    return FC


def extract_feature(data):
    feat = []
    for i in range(data.shape[0]):
        fc = Persion_matrix(data[i])
        fc = pd.DataFrame(fc)
        fc = fc.fillna(0)
        feat.append(fc.values)
    return feat


def FC_adj(data):
    feature = extract_feature(data)
    feature = np.array(feature)
    return feature


def FC_adj2(data1, data2):
    feature = []
    for i in range(data1.shape[0]):
        fc = np.corrcoef(data1[i], data2[i], rowvar=0)
        fc = pd.DataFrame(fc)
        fc = fc.fillna(0)
        feature.append(fc.values)
    feature = np.array(feature)
    return feature


def get_flops_params():
    sess = tf.compat.v1.Session()
    graph = sess.graph
    flops = tf.compat.v1.profiler.profile(graph, options=tf.compat.v1.profiler.ProfileOptionBuilder.float_operation())
    params = tf.compat.v1.profiler.profile(graph, options=tf.compat.v1.profiler.ProfileOptionBuilder.trainable_variables_parameter())
    print('FLOPs: {};    Trainable params: {}'.format(flops.total_float_ops, params.total_parameters))



def main():

    crops = 10
    ###### load dataset
    ASD_feats = np.load('Data/ADHD/ADHD_feats_norm.npy', allow_pickle=True)
    ASD_labels = np.load('Data/ADHD/ADHD_labels_norm.npy')
    ASD_phenos = np.load('Data/ADHD/ADHD_phenos_norm.npy', allow_pickle=True)

    # ASD_feats = np.load('Data/ASD/ASD_feats_norm.npy', allow_pickle=True)
    # ASD_labels = np.load('Data/ASD/ASD_labels_norm.npy')
    # ASD_phenos = np.load('Data/ASD/ASD_phenos_norm.npy', allow_pickle=True)

    # ASD_feats = np.load('Data/ABIDE2/ABIDEII_data_global.npy', allow_pickle=True)
    # ASD_labels = np.load('Data/ABIDE2/ABIDEII_labels_global.npy')
    # ASD_phenos = np.load('Data/ABIDE2/ABIDEII_phenos_global.npy', allow_pickle=True)

    augment_ASD_feats, augment_ASD_labels, augment_ASD_pheno, sk_data, sk_label = data_augment(ASD_feats, ASD_labels,
                                                                                               ASD_phenos, crops)

    print(ASD_feats.shape)
    ####### Cross validation
    CV_count = 10
    ASD_mean = []
    ASD_sub_mean = []
    ASD_sub2_mean = []

    ASD_sensitive_mean = []
    ASD_specificity_mean = []
    ASD_auc_mean = []

    for i_count in range(CV_count):

        kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=2 ** (i_count + 1))
        # kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=2 ** (1 + 1))

        j_count = 0

        ADHD_acc = []
        ADHD_sub_acc = []
        ADHD_sub2_acc = []

        ADHD_sensitive = []
        ADHD_specificity = []
        ADHD_auc = []

        for train_index, test_index in kf.split(sk_data, sk_label):

            # j_count += 1
            #
            # if j_count != 9:
            #     continue
            #
            # print(j_count)

            aug_x_train_list = []
            aug_x_test_list = []
            true_test_label_list = []
            count = 0

            for i in range(sk_data.shape[0]):
                if i in train_index:
                    for k in range(crops):
                        aug_x_train_list.append(count)
                        count = count + 1

                else:
                    true_test_label_list.append(i)
                    for k in range(crops):
                        aug_x_test_list.append(count)
                        count = count + 1

            # training data
            x_train = augment_ASD_feats[aug_x_train_list]
            y_train = augment_ASD_labels[aug_x_train_list]
            x_train_pheno = augment_ASD_pheno[aug_x_train_list]

            print(x_train.shape)

            node_num, feature_dim = x_train.shape[2], x_train.shape[1]

            # testing data
            x_test = augment_ASD_feats[aug_x_test_list]
            y_test = augment_ASD_labels[aug_x_test_list]
            x_test_pheno = augment_ASD_pheno[aug_x_test_list]

            true_test_label = sk_label[true_test_label_list]

            Graph_connect = FC_adj(sk_data[train_index])
            Graph_connect = Graph_connect.mean(axis=0)
            for i in range(Graph_connect.shape[0]):
                for j in range(Graph_connect.shape[1]):
                    if Graph_connect[i][j] > 0.1:
                        Graph_connect[i][j] = 1
                    else:
                        Graph_connect[i][j] = 0

            zero_padding = np.zeros(shape=(node_num, node_num))
            temporal_connect = np.eye(node_num)

            A_Gtrain = np.repeat(Graph_connect[None, :], x_train.shape[0], axis=0)
            A_Gtest = np.repeat(Graph_connect[None, :], x_test.shape[0], axis=0)

            A_Ttrain = np.repeat(temporal_connect[None, :], x_train.shape[0], axis=0)
            A_Ttest = np.repeat(temporal_connect[None, :], x_test.shape[0], axis=0)

            A_Ptrain = np.repeat(zero_padding[None, :], x_train.shape[0], axis=0)
            A_Ptest = np.repeat(zero_padding[None, :], x_test.shape[0], axis=0)

            # shuffle
            index = [i for i in range(x_train.shape[0])]
            random.shuffle(index)
            A_Gtrain = A_Gtrain[index]
            A_Ttrain = A_Ttrain[index]
            A_Ptrain = A_Ptrain[index]

            x_train_pheno = x_train_pheno[index]
            y_train = y_train[index]
            x_test_pheno = x_test_pheno.reshape(-1, 4)

            x_train = x_train[index].swapaxes(2, 1)

            model = STIGR_ADHD(feature_dim, node_num)

            model.summary()

            reduce_lr = ReduceLROnPlateau(monitor='val_acc', factor=0.2,
                                          patience=5, min_lr=0.00001)
            early_stopping = EarlyStopping(monitor='output1_accuracy', patience=20, verbose=2)

            model.fit(x=[x_train, A_Gtrain, A_Ttrain, A_Ptrain, x_train_pheno], y=[y_train,A_Ttrain], epochs=50, batch_size=64,
                      validation_split=0.05, callbacks=[early_stopping])

            x_test = x_test.swapaxes(2, 1)
            ADHD_predict = model.predict(x=[x_test, A_Gtest, A_Ttest, A_Ptest, x_test_pheno])

            ADHD_pred = [1 if prob >= 0.5 else 0 for prob in ADHD_predict[0]]

            pred = []
            pred2 = []
            true_labels = []

            true_score = []
            pred_score = []
            for i in range(len(test_index)):
                ADHD_subject = ADHD_pred[i * crops:i * crops + crops]
                if np.count_nonzero(ADHD_subject) >= crops / 2:
                    pred.append(1)
                else:
                    pred.append(0)

                ADHD_subject_score = ADHD_predict[0][i * crops:i * crops + crops]
                pred_score.append(np.sum(ADHD_subject_score) / crops)
                if np.sum(ADHD_subject_score) / crops >= 0.5:
                    pred2.append(1)
                else:
                    pred2.append(0)

                ADHD_true_subject = y_test[i * crops:i * crops + crops]
                if np.count_nonzero(ADHD_true_subject) >= crops / 2:
                    true_labels.append(1)
                else:
                    true_labels.append(0)


            [[TN, FP], [FN, TP]] = confusion_matrix(true_test_label, pred).astype(float)
            specificity = TN / (FP + TN)
            sensivity = recall = TP / (TP + FN)
            auc = roc_auc_score(true_test_label, pred_score)

            ADHD_sub_acc.append(accuracy_score(true_test_label, pred))
            ADHD_sub2_acc.append(accuracy_score(true_test_label, pred2))
            ADHD_auc.append(auc)
            ADHD_specificity.append(specificity)
            ADHD_sensitive.append(sensivity)

            print("ADHD accuracy: " + str(metrics.accuracy_score(y_test, ADHD_pred)))
            print("ADHD sub accuracy: " + str(accuracy_score(true_test_label, pred)))
            print("ADHD sub2 accuracy: " + str(accuracy_score(true_test_label, pred2)))
            print("ADHD sub2 auc: " + str(auc))
            print("ADHD sub2 specificity: " + str(specificity))
            print("ADHD sub2 sensivity: " + str(sensivity))

            K.clear_session()
            # tf.reset_default_graph()
            del [A_Gtrain, A_Ptrain, A_Ttrain, A_Gtest, A_Ttest, A_Ptest, Graph_connect, temporal_connect, zero_padding]
            gc.collect()

        ASD_sub_mean.append(np.mean(ADHD_sub_acc))
        ASD_sub2_mean.append(np.mean(ADHD_sub2_acc))
        ASD_auc_mean.append(np.mean(ADHD_auc))
        ASD_specificity_mean.append(np.mean(ADHD_specificity))
        ASD_sensitive_mean.append(np.mean(ADHD_sensitive))

        print("ADHD_sub_bert_mean:" + str(ASD_sub_mean))
        print("ADHD_sub2 mean: " + str(ASD_sub2_mean))
        print("ADHD_sub2 auc_mean: " + str(ASD_auc_mean))
        print("ADHD_sub2 specificity_mean: " + str(ASD_specificity_mean))
        print("ADHD_sub2 sensitive_meann: " + str(ASD_sensitive_mean))


if __name__ == '__main__':

    main()