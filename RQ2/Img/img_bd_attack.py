# ART
# # coding=gbk
from __future__ import absolute_import, division, print_function, unicode_literals

#import tensorflow as tf
#tf.compat.v1.disable_eager_execution()

import os, pickle, random, sys

# 获取项目的根目录
project_root = '/root/zx/Thesis_Code/'

# 将项目根目录添加到 sys.path 中
sys.path.append(project_root)


import argparse

from os.path import abspath

module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)

import warnings

warnings.filterwarnings('ignore')
#import tensorflow.keras.backend as k
#from tensorflow.keras.models import Sequential
#from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Activation, Dropout
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#from mpl_toolkits import mplot3d



def parse_arguments():
    parser = argparse.ArgumentParser(description='AC PARAMs')

    # 添加命令行参数
    parser.add_argument('--TRIGGER_TYPE', type=str, default='badnet')
    parser.add_argument('--GPU_NUM', type=str, default='0')
    parser.add_argument('--POS_RATE', type=float, default=0.1)
    parser.add_argument('--MODEL_NAME', type=str, default='VGG16')
    parser.add_argument('--EPOCH', type=int, default=1)

    return parser.parse_args()

# model evaluation
def evaluation(model,X_test, Y_test,mods,lbl,snrs,train_idx,test_idx):
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


def load_img_pickle(root_path, pickle_name):
    with open(root_path + pickle_name, 'rb') as f:
        pickle_data = pickle.load(f)
    pickle_data = np.array(pickle_data).astype('float32')
    pickle_data /= 255
    return pickle_data

def load_label_pickle(root_path, pickle_name):
    with open(root_path + pickle_name, 'rb') as f:
        pickle_data = pickle.load(f)
    return pickle_data

def load_img_datasets(root_path):
    
    benign_path = '/root/zx/Thesis_Code/datasets/constellation_benign/'
    
    X_train_badnet = load_img_pickle(root_path,pickle_name = 'X_train_badnet.pkl')
    X_test_badnet = load_img_pickle(root_path,pickle_name = 'X_test_badnet.pkl')
    X_test_benign = load_img_pickle(benign_path,pickle_name = 'X_test_badnet.pkl')
    
    Y_train_badnet = load_label_pickle(root_path,pickle_name = 'Y_train_badnet.pkl')
    Y_test_badnet = load_label_pickle(root_path,pickle_name = 'Y_test_badnet.pkl')
    Y_test_benign = load_label_pickle(benign_path,pickle_name = 'Y_test_badnet.pkl')
    
    return X_train_badnet,Y_train_badnet,X_test_badnet,Y_test_badnet,X_test_benign,Y_test_benign

def main():

    args = parse_arguments()
    
    # get mods,lbl,snrs,train_idx,test_idx
    from trigger_config import load_data
    X_train,X_test,Y_train,Y_test,mods,lbl,snrs,train_idx,test_idx = load_data()

    # load img
    root_path = '/root/zx/Thesis_Code/datasets/constellation_' + args.TRIGGER_TYPE +'/'

    
    X_train_badnet,Y_train_badnet,X_test_badnet,Y_test_badnet,X_test_benign,Y_test_benign = load_img_datasets(root_path)
    
    from sklearn.model_selection import train_test_split
    print(X_train_badnet.shape)
    print(Y_train_badnet.shape)
    x_train, x_val, y_train, y_val = train_test_split(X_train_badnet, Y_train_badnet, test_size=0.111, random_state=42)
    
    # build_model
    from mltools import build_model
    
    num_classes = 11
    target_DNN = args.MODEL_NAME
    model = build_model(target_model=target_DNN)
    
    # train
    from mltools import train
    
    model, history = train(model, x_train, y_train, x_val, y_val, nb_epoch=args.EPOCH, batch_size=1024)
    
    root_path = '/root/zx/Thesis_Code/dl_amc_backdoor/constellation/'
    pos_model_name = f"{args.MODEL_NAME}_{args.TRIGGER_TYPE}_{args.POS_RATE}_{args.EPOCH}"
    
    model.save(root_path + 'saved_model/' + pos_model_name + '.h5')
    
    # evaluation
    # CA
    print('CA')
    CA = evaluation(model, X_test_benign, Y_test_benign,mods,lbl,snrs,train_idx,test_idx)
    # print('ASR')
    ASR = evaluation(model, X_test_badnet, Y_test_badnet,mods,lbl,snrs,train_idx,test_idx)
    
    # save results
    import pandas as pd
    
    # writing to Excel
    writer = pd.ExcelWriter(root_path + 'results/' + pos_model_name + '.xlsx')
    df1 = pd.DataFrame(CA)
    df2 = pd.DataFrame(ASR)
    df = pd.concat([df1, df2], axis=1)
    df.to_excel(writer, 'CA_ASR', float_format='%.5f')
    
    writer.save()
    
    writer.close()

if __name__ == '__main__':

    import time
    
    st = time.time()
    
    main()
    
    et = time.time()
    
    print('time cost', et-st, 's')

