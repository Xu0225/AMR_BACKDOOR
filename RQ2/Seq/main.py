# -*- coding:utf-8 -*-
import argparse
import warnings

warnings.filterwarnings('ignore')

import os, pickle, random, sys


# 获取项目的根目录
project_root = '/root/zx/Thesis_Code/'

# 将项目根目录添加到 sys.path 中
sys.path.append(project_root)

import numpy as np
import copy
import tensorflow.keras.backend as K

from tensorflow import keras
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.keras.regularizers import *
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import model_from_json

from sklearn.model_selection import train_test_split

from mltools import build_model
from mltools import fix_dim

# from rmlmodel.Sequence.CLDNN import CLDNNLikeModel
# from rmlmodel.Sequence.ResNet import ResNetLikeModel
# from rmlmodel.Sequence.VGG import VGGLikeModel


from rmlmodel.Sequence.vtcnn2 import VTCNN2
from rmlmodel.Sequence.CNN2 import CNN2
from rmlmodel.Sequence.CNN2Model import CNN2Model

from rmlmodel.Sequence.CLDNNLikeModel import CLDNNLikeModel
from rmlmodel.Sequence.CLDNNLikeModel1 import CLDNNLikeModel1
from rmlmodel.Sequence.CLDNNLikeModel2 import CLDNNLikeModel2

# -*- coding:utf-8 -*-
from rmlmodel.Sequence.CGDNN import CGDNN
from rmlmodel.Sequence.CuDNNLSTMModel import LSTMModel
from rmlmodel.Sequence.DAE import DAE
from rmlmodel.Sequence.DCNNPF import DCNNPF
from rmlmodel.Sequence.DenseNet import DenseNet
from rmlmodel.Sequence.GRUModel import GRUModel
from rmlmodel.Sequence.ICAMC import ICAMC
from rmlmodel.Sequence.MCLDNN import MCLDNN
from rmlmodel.Sequence.MCNET import MCNET
from rmlmodel.Sequence.PETCGDNN import PETCGDNN
from rmlmodel.Sequence.ResNet import ResNet

from trigger_config import load_data
from trigger_config import set_trigger_config

from plot_tools import plot_signal
from plot_tools import plot_constellation
from mltools import get_seq_data

def parse_arguments():
    parser = argparse.ArgumentParser(description='Process trigger configurations.')
    parser.add_argument('--TRIGGER_TYPE', type=str, default = 'badnet', help='Type of trigger (badnet, random_location, hanning, spectrum_shift, phase_shift, remapped_awgn)')
    parser.add_argument('--POS_RATE', type=float, default=0.1, help='Positive rate of samples to be injected with the trigger.')
    parser.add_argument('--DATA_TYPE', type=str, default='train', help='Type of data (train or test).')
    parser.add_argument('--GPU_NUM', type=str, default='1', help='GPU NUM')
    parser.add_argument('--EPOCH', type=int, default=5)
    parser.add_argument('--MODEL_NAME', type=str, default='CNN2')
    parser.add_argument('--REP', type=str, default='AP')

    return parser.parse_args()

# model evaluation
def evaluation(model,X_test, Y_test):
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

if __name__ == '__main__':

    args = parse_arguments()

    EPOCH = args.EPOCH
    MODEL_NAME = args.MODEL_NAME
    POS_RATE = args.POS_RATE
    TRIGGER_TYPE = args.TRIGGER_TYPE
    REP = args.REP

    #os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    #os.environ["CUDA_VISIBLE_DEVICES"] = args.GPU_NUM

    X_train,X_test,Y_train,Y_test,mods,lbl,snrs,train_idx,test_idx = load_data()


    X_train_modified, Y_train_modified = set_trigger_config(X_train.copy(), Y_train.copy(), pos_rate=args.POS_RATE,
                                                            trigger_type=args.TRIGGER_TYPE, data_type='train')
                                                            
    # REP = IQ/AP/FFT
    X_train_modified = get_seq_data(X_train_modified, seq_dtype = args.REP)
    
    X_train_modified = np.nan_to_num(X_train_modified)
    
    x_train, x_val, y_train, y_val = train_test_split(X_train_modified, Y_train_modified, test_size=0.111, random_state=42)


    ROOT_PATH = '/root/zx/Thesis_Code/dl_amc_backdoor/all_to_one/'
    #ROOT_PATH = 'D:/zhaixu/Thesis_Code/dl_amc_backdoor/all_to_one/'
    #FILE_NAME = MODEL_NAME + '_' + TRIGGER_TYPE + '_' + str(POS_RATE) + '_' + str(EPOCH)
    FILE_NAME = f"{REP}_{MODEL_NAME}_{TRIGGER_TYPE}_{POS_RATE}_{EPOCH}"
    EXCEL_NAME = f"{REP}_{MODEL_NAME}_{TRIGGER_TYPE}"

    # train
    from mltools import train
    import pandas as pd

    # 创建 Excel writer
    result_file_path = ROOT_PATH + 'results/' + EXCEL_NAME + '.xlsx'
    writer = pd.ExcelWriter(result_file_path)

    all_results = pd.DataFrame()  # 创建一个用于保存所有结果的DataFrame

    # model training
    # model_TBD = ['VTCNN2',
    #              'CNN2',
    #              'CNN2Model',
    #              'CLDNNLikeModel',
    #              'CLDNNLikeModel1',
    #              'CLDNNLikeModel2',
    #              'CGDNN',
    #              'LSTMModel',
    #              'DAE',
    #              #'DCNNPF',
    #              'DenseNet',
    #              'GRUModel',
    #              'ICAMC',
    #              #'MCLDNN',
    #              'MCNET',
    #              #'PETCGDNN',
    #              'ResNet',
    #             ]
    model_TBD = [MODEL_NAME]

    for target_DNN in model_TBD:
        # build_model
        model = build_model(target_model=target_DNN)

        # reshape input
        input_shape = model.get_input_shape_at(0)
        x_train = x_train.reshape((x_train.shape[0],) + tuple(input_shape[1:]))
        x_val = x_val.reshape((x_val.shape[0],) + tuple(input_shape[1:]))

        # train_model
        # model,history = train(model,X_train_attacked, Y_train_attacked, X_val_attacked, Y_val_attacked,nb_epoch = 20,batch_size = 1024)
        model, history = train(model, x_train, y_train, x_val, y_val, nb_epoch=EPOCH, batch_size=1024)

        # save model

        save_name = FILE_NAME + '.h5'
        model.save(ROOT_PATH + 'saved_model/' + save_name)

        # CA
        print('CA')
        X_test_CA = copy.deepcopy(X_test)
        
        X_test_CA = get_seq_data(X_test_CA, seq_dtype = args.REP)
        
        X_test_CA = X_test_CA.reshape((X_test_CA.shape[0],) + tuple(input_shape[1:]))
        print(X_test_CA.shape)
        CA = evaluation(model, X_test_CA, Y_test)

        # ASR
        print('ASR')
        X_test_ASR, Y_test_ASR = set_trigger_config(X_test.copy(),Y_test.copy(),pos_rate=args.POS_RATE,
                                               trigger_type=args.TRIGGER_TYPE, data_type='test')
                                               
        X_test_ASR = get_seq_data(X_test_ASR, seq_dtype = args.REP)

        # X_test = fix_dim(X_test)
        X_test_ASR = X_test_ASR.reshape((X_test_ASR.shape[0],) + tuple(input_shape[1:]))
        print(X_test_ASR.shape)
        ASR = evaluation(model, X_test_ASR, Y_test_ASR)

        # writing to Excel
        CA_result = pd.DataFrame(CA, columns=[target_DNN + '_CA'])
        ASR_result = pd.DataFrame(ASR, columns=[target_DNN + '_ASR'])
        model_results = pd.concat([CA_result, ASR_result], axis=1)

        all_results = pd.concat([all_results, model_results], axis=1)

    # 将 all_results 保存到 Excel writer
    all_results.to_excel(writer, sheet_name=EXCEL_NAME, float_format='%.5f', index=False)

    writer.save()

    writer.close()