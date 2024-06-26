# ART
# # coding=gbk
import os, pickle, random, sys
sys.path.append('D:\\zhaixu\\Thesis_Code\\dl_amc_defense_seq')
sys.path.append('D:\\zhaixu\\Thesis_Code')

# log_print = open('Defalust.log', 'w')
# sys.stdout = log_print
# sys.stderr = log_print

import warnings
import copy
warnings.filterwarnings('ignore')
import numpy as np
import argparse
from matplotlib import pyplot as plt
import tensorflow as tf
tf.compat.v1.disable_eager_execution()

from art.estimators.classification import KerasClassifier
from art.attacks.poisoning import PoisoningAttackBackdoor
from art.attacks.poisoning.perturbations import add_pattern_bd, add_single_bd, insert_image
from art.utils import load_mnist, preprocess
from art.defences.detector.poison import ActivationDefence
from art.defences.transformer.poisoning import NeuralCleanse
from art.estimators.poison_mitigation import KerasNeuralCleanse
from sklearn.model_selection import train_test_split

from mltools import evaluation
from trigger_config import load_data
from trigger_config import set_trigger_config

def parse_arguments():
    parser = argparse.ArgumentParser(description='AC PARAMs')

    # 添加命令行参数
    parser.add_argument('--TRIGGER_TYPE', type=str, default='spectrum_shift')
    parser.add_argument('--GPU_NUM', type=str, default='0')
    parser.add_argument('--POS_RATE', type=float, default=0.1)
    parser.add_argument('--MODEL_NAME', type=str, default='CNN2')
    parser.add_argument('--EPOCH', type=int, default=100)

    return parser.parse_args()

def compute_TPR_FPR(y_true, y_pred):
    from sklearn.metrics import confusion_matrix

    # 计算混淆矩阵
    conf_matrix = confusion_matrix(y_true, y_pred)

    # 提取混淆矩阵的各个元素
    TP = conf_matrix[0, 0]  # True Positives
    FN = conf_matrix[0, 1]  # False Negatives
    FP = conf_matrix[1, 0]  # False Positives
    TN = conf_matrix[1, 1]  # True Negatives

    # 计算 True Positive Rate (TPR) 和 False Positive Rate (FPR)
    TPR = TP / (TP + FN)
    FPR = FP / (FP + TN)

    print(f'True Positive Rate (TPR): {TPR:.4f}')
    print(f'False Positive Rate (FPR): {FPR:.4f}')

    return TPR, FPR

def metrics(y_true, y_pred):

    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

    # 计算混淆矩阵
    conf_matrix = confusion_matrix(y_true, y_pred)

    # 计算准确率
    accuracy = accuracy_score(y_true, y_pred)

    # 计算精确度
    precision = precision_score(y_true, y_pred,pos_label=0)

    # 计算召回率
    recall = recall_score(y_true, y_pred,pos_label=0)

    # 计算 F1 分数
    f1 = f1_score(y_true, y_pred,pos_label=0)

    print("Confusion Matrix:")
    print(conf_matrix)
    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1 Score:", f1)

# model evaluation
def evaluation(model,X_test, Y_test,test_idx,mods,snrs,lbl):
    classes = mods
    acc = []
    for snr in snrs:

        # extract classes @ SNR
        test_SNRs = list(map(lambda x: lbl[x][1], test_idx))
        test_X_i = X_test[np.where(np.array(test_SNRs) == snr)]
        test_Y_i = Y_test[np.where(np.array(test_SNRs) == snr)]

        # estimate classes
        test_Y_i_hat = model.predict(test_X_i)
        conf = np.zeros([len(classes), len(classes)])

        for i in range(0, test_X_i.shape[0]):
            j = list(test_Y_i[i, :]).index(1)
            k = int(np.argmax(test_Y_i_hat[i, :]))
            conf[j, k] = conf[j, k] + 1

        cor = np.sum(np.diag(conf))
        ncor = np.sum(conf) - cor
        #print("Overall Accuracy: ", cor / (cor+ncor))
        print("snr:",snr,"acc:",cor / (cor + ncor))
        acc.append(1.0 * cor / (cor + ncor))
    acc_mean = sum(acc) / len(acc)
    print('acc_mean: ',acc_mean)
    acc.append(acc_mean)
    return acc




def main():

    args = parse_arguments()

    X_train,X_test,Y_train,Y_test,mods,lbl,snrs,train_idx,test_idx = load_data()

    X_train_modified, Y_train_modified = set_trigger_config(X_train.copy(), Y_train.copy(), pos_rate=args.POS_RATE,
                                                            trigger_type=args.TRIGGER_TYPE, data_type='train')

    x_train, x_val, y_train, y_val = train_test_split(X_train_modified, Y_train_modified, test_size=0.111,
                                                      random_state=42)

    # load model
    from tensorflow.keras.models import load_model
    #poisoned_model = load_model('D:/zhaixu/Thesis_Code/dl_amc_backdoor/all_to_one/saved_model/posioned_'+ args.MODEL_NAME + '_Hanning_EPOCH_100_POS_RATE_0.1.h5')

    root_path = 'D:/zhaixu/Thesis_Code/dl_amc_backdoor/all_to_one/saved_model/'
    pos_model_name = f"{args.MODEL_NAME}_{args.TRIGGER_TYPE}_{args.POS_RATE}_{args.EPOCH}"
    poisoned_model = load_model(root_path + pos_model_name + '.h5')
    #poisoned_model = load_model(root_path + 'CNN2_spectrum_shift_0.1_100.h5')
    poisoned_model.summary()

    # reshape train input
    input_shape = poisoned_model.get_input_shape_at(0)
    X_train = X_train.reshape((X_train.shape[0],) + tuple(input_shape[1:]))
    X_train_modified = X_train_modified.reshape((X_train_modified.shape[0],) + tuple(input_shape[1:]))
    x_train = x_train.reshape((x_train.shape[0],) + tuple(input_shape[1:]))
    x_val = x_val.reshape((x_val.shape[0],) + tuple(input_shape[1:]))

    # detect
    # detect
    from art.defences.detector.poison import SpectralSignatureDefense
    from art.defences.transformer.poisoning import STRIP
    from art.defences.detector.poison import ActivationDefence
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

    classifier = KerasClassifier(model=poisoned_model)

    defence = SpectralSignatureDefense(classifier, X_train_modified, Y_train_modified)
    report, is_clean_lst = defence.detect_poison(nb_clusters=2, nb_dims=11, reduce="PCA")

    # 获取第7个位置的值
    index = 7
    y_true = np.array([0 if sample[index] == 1 else 1 for sample in Y_train_modified])

    y_pred = is_clean_lst
    y_pred = np.array(is_clean_lst)

    # detect eval
    compute_TPR_FPR(y_true, y_pred)
    metrics(y_true, y_pred)


    # # model eval asr/ca
    #
    # no defense
    print('No defense eval\n')
    X_test_benign = X_test.reshape((X_test.shape[0],) + tuple(input_shape[1:]))

    X_test_modified, Y_test_modified = set_trigger_config(X_test.copy(), Y_test.copy(), pos_rate=args.POS_RATE,
                                                            trigger_type=args.TRIGGER_TYPE, data_type='test')

    X_test_modified = X_test_modified.reshape((X_test_modified.shape[0],) + tuple(input_shape[1:]))
    #
    print('CA')
    acc_ca_no_defense = evaluation(poisoned_model,X_test_benign,Y_test,test_idx,mods,snrs,lbl)
    print('ASR')
    acc_asr_no_defense = evaluation(poisoned_model,X_test_modified,Y_test_modified,test_idx,mods,snrs,lbl)


    # filter data
    is_clean_lst = np.array(is_clean_lst)
    clean_indices = np.where(is_clean_lst == 1)[0]
    cleaned_x = X_train_modified[clean_indices]
    cleaned_y = Y_train_modified[clean_indices]

    x_train, x_val, y_train, y_val = train_test_split(cleaned_x, cleaned_y, test_size=0.111,
                                                      random_state=42)
    # retrain model
    from mltools import build_model, train
    poisoned_model = build_model(target_model=args.MODEL_NAME)
    poisoned_model, history = train(poisoned_model, x_train, y_train, x_val, y_val, nb_epoch=args.EPOCH, batch_size=1024)


    # eval repaired model
    print('Repaired model eval\n')
    print('CA')
    acc_ca_repaired = evaluation(poisoned_model,X_test_benign,Y_test,test_idx,mods,snrs,lbl)
    print('ASR')
    acc_asr_repaired = evaluation(poisoned_model,X_test_modified,Y_test_modified,test_idx,mods,snrs,lbl)

if __name__ == '__main__':
    import time
    start_time = time.time()

    main()

    end_time = time.time()
    print('time spend: ',end_time - start_time,'s')
