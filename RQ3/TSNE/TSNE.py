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
    parser.add_argument('--TRIGGER_TYPE', type=str, default='phase_shift')
    parser.add_argument('--GPU_NUM', type=str, default='0')
    parser.add_argument('--POS_RATE', type=float, default=0.1)
    parser.add_argument('--MODEL_NAME', type=str, default='CNN2')
    parser.add_argument('--EPOCH', type=int, default=0)

    return parser.parse_args()

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

if __name__ == '__main__':
    import time
    start_time = time.time()

    main()

    end_time = time.time()
    print('time spend: ', end_time - start_time, 's')
