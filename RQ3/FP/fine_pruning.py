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
import mltools

from art.estimators.classification import KerasClassifier
from art.attacks.poisoning import PoisoningAttackBackdoor
from art.attacks.poisoning.perturbations import add_pattern_bd, add_single_bd, insert_image
from art.utils import load_mnist, preprocess
from art.defences.detector.poison import ActivationDefence
from art.defences.transformer.poisoning import NeuralCleanse
from art.estimators.poison_mitigation import KerasNeuralCleanse
from sklearn.model_selection import train_test_split

from trigger_config import load_data
from trigger_config import set_trigger_config

from tensorflow import keras
from sklearn.preprocessing import normalize  # Adjust based on your preprocessing needs

def parse_arguments():
    parser = argparse.ArgumentParser(description='AC PARAMs')

    # 添加命令行参数
    parser.add_argument('--TRIGGER_TYPE', type=str, default='phase_shift')
    parser.add_argument('--REP', type=str, default='IQ')
    parser.add_argument('--POS_RATE', type=float, default=0.1)
    parser.add_argument('--MODEL_NAME', type=str, default='CNN2')
    parser.add_argument('--EPOCH', type=int, default=2)

    return parser.parse_args()

def get_conv_index(model):
    convindex = []
    for i, layer in enumerate(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D) or isinstance(layer, tf.keras.layers.Conv1D):
            convindex.append(i)
    return convindex

def clear_max_weights(weights, thresh):
    for i in range(len(weights)):
        for j in range(len(weights[i])):
            for k in range(len(weights[i][j])):
                for m in range(len(weights[i][j][k])):
                    if weights[i][j][k][m] > thresh:
                        weights[i][j][k][m] = 0
    return weights

def calc_top_X_percent_weight(weights, fraction):
    flat_weights = weights.flatten()
    num_weights_to_keep = int(len(flat_weights) * (1 - fraction))
    top_weights = np.sort(flat_weights)[-num_weights_to_keep]
    return top_weights

def fineprune(model, pruning_fraction):
    layer_weights = []
    convindex = get_conv_index(model)
    for i in convindex:
        layer_weights.append(model.layers[i].get_weights()[0])

    max_weights_thr = []
    for weights in layer_weights:
        max_weights_thr.append(calc_top_X_percent_weight(weights, pruning_fraction))

    new_weights = []
    for i, weights in enumerate(layer_weights):
        new_weights.append(clear_max_weights(weights, max_weights_thr[i]))

    for i, idx in enumerate(convindex):
        current_weights = model.layers[idx].get_weights()
        current_weights[0] = new_weights[i]
        model.layers[idx].set_weights(current_weights)

    return model

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
        #print("snr:",snr,"acc:",cor / (cor + ncor))
        acc.append(1.0 * cor / (cor + ncor))
    acc_mean = sum(acc) / len(acc)
    print('acc_mean: ',acc_mean)
    acc.append(acc_mean)
    return acc


def incremental_pruning(model, X_train, Y_train, X_val, Y_val,
                        initial_fraction=0.1,
                        final_fraction=0.9,
                        steps=5,
                        epochs_per_step=5):
    pruning_fractions = np.linspace(initial_fraction, final_fraction, steps)
    clean_accuracies = []
    attack_sucecess_rates = []
    for fraction in pruning_fractions:
        model = fineprune(model, fraction, incremental=True)
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        model.fit(X_train, Y_train, validation_data=(X_val, Y_val), epochs=epochs_per_step, verbose=0)

        print('CA')
        acc_ca_no_defense = evaluation(model, X_test_benign, Y_test, test_idx, mods, snrs, lbl)
        print('ASR')
        acc_asr_no_defense = evaluation(model, X_test_modified, Y_test_modified, test_idx, mods, snrs, lbl)

        clean_accuracies.append(acc_ca_no_defense)
        attack_sucecess_rates.append(acc_asr_no_defense)
        print(f'Pruning Fraction: {fraction}, CA: {acc_ca_no_defense}, ASR: {acc_asr_no_defense}')
    # plt.plot(pruning_fractions, clean_accuracies, attack_sucecess_rates)
    # plt.xlabel('Pruning Fraction')
    # plt.ylabel('CA_ASR')
    # plt.title('Model Performance vs Pruning Fraction')
    # plt.show()

    return model

if __name__ == '__main__':
    args = parse_arguments()
    # load data
    X_train, X_test, Y_train, Y_test, mods, lbl, snrs, train_idx, test_idx = load_data()

    X_train_modified, Y_train_modified = set_trigger_config(X_train.copy(), Y_train.copy(), pos_rate=0.1,
                                                            trigger_type=args.TRIGGER_TYPE, data_type='train')

    x_train, x_val, y_train, y_val = train_test_split(X_train_modified, Y_train_modified, test_size=0.111,
                                                      random_state=42)

    # load model
    from tensorflow.keras.models import load_model

    # poisoned_model = load_model('D:/zhaixu/Thesis_Code/dl_amc_backdoor/all_to_one/saved_model/posioned_'+ args.MODEL_NAME + '_Hanning_EPOCH_100_POS_RATE_0.1.h5')

    root_path = 'D:/zhaixu/Thesis_Code/dl_amc_backdoor/all_to_one/saved_model/'
    pos_model_name = f"{args.REP}_{args.MODEL_NAME}_{args.TRIGGER_TYPE}_{args.POS_RATE}_{args.EPOCH}"
    poisoned_model = load_model(root_path + pos_model_name + '.h5')


    # reshape train input
    input_shape = poisoned_model.get_input_shape_at(0)
    X_train = X_train.reshape((X_train.shape[0],) + tuple(input_shape[1:]))
    X_train_modified = X_train_modified.reshape((X_train_modified.shape[0],) + tuple(input_shape[1:]))
    x_train = x_train.reshape((x_train.shape[0],) + tuple(input_shape[1:]))
    x_val = x_val.reshape((x_val.shape[0],) + tuple(input_shape[1:]))

    # no defense
    print('No defense eval\n')
    X_test_benign = X_test.reshape((X_test.shape[0],) + tuple(input_shape[1:]))
    X_test_modified, Y_test_modified = set_trigger_config(X_test.copy(), Y_test.copy(), pos_rate=0.1,
                                                          trigger_type=args.TRIGGER_TYPE, data_type='test')
    X_test_modified = X_test_modified.reshape((X_test_modified.shape[0],) + tuple(input_shape[1:]))
    print('CA')
    acc_ca_no_defense = evaluation(poisoned_model, X_test_benign, Y_test, test_idx, mods, snrs, lbl)
    print('ASR')
    acc_asr_no_defense = evaluation(poisoned_model, X_test_modified, Y_test_modified, test_idx, mods, snrs, lbl)

    # 干净数据集微调
    x_train_ca, x_val_ca, y_train_ca, y_val_ca = train_test_split(X_train, Y_train, test_size=0.111,
                                                                  random_state=42)
    # Incremental pruning 阈值调节实验
    # model = incremental_pruning(poisoned_model, x_train, y_train, x_val, y_val)

    model = fineprune(poisoned_model, pruning_fraction=0.1)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(x_train_ca, y_train_ca, validation_data=(x_val_ca, y_val_ca), epochs=2, verbose=0)

    print('CA')
    acc_ca_no_defense = evaluation(model, X_test_benign, Y_test, test_idx, mods, snrs, lbl)
    print('ASR')
    acc_asr_no_defense = evaluation(model, X_test_modified, Y_test_modified, test_idx, mods, snrs, lbl)
