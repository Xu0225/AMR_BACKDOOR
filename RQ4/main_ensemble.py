import argparse
import warnings

warnings.filterwarnings('ignore')

import pickle, random, sys

sys.path.append('D:\\zhaixu\\Thesis_Code')

import os, pickle, random, sys

import numpy as np
import copy

from sklearn.model_selection import train_test_split

from mltools import build_model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from trigger_config import load_data
from trigger_config import set_trigger_config
from tensorflow.keras.models import load_model
from mltools import train
def parse_arguments():
    parser = argparse.ArgumentParser(description='Process trigger configurations.')
    parser.add_argument('--TRIGGER_TYPE', type=str, default = 'benign', help='Type of trigger (badnet, random_location, hanning, spectrum_shift, phase_shift, remapped_awgn)')
    parser.add_argument('--POS_RATE', type=float, default=0.1, help='Positive rate of samples to be injected with the trigger.')
    parser.add_argument('--DATA_TYPE', type=str, default='train', help='Type of data (train or test).')
    parser.add_argument('--GPU_NUM', type=str, default='0', help='GPU NUM')
    parser.add_argument('--EPOCH', type=int, default=100)
    parser.add_argument('--MODEL_NAME', type=str, default='CNN2')

    return parser.parse_args()


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
    benign_path = 'D:/zhaixu/Thesis_Code/datasets/constellation_benign/'

    X_train_badnet = load_img_pickle(root_path, pickle_name='X_train_badnet.pkl')
    X_test_badnet = load_img_pickle(root_path, pickle_name='X_test_badnet.pkl')
    X_test_benign = load_img_pickle(benign_path, pickle_name='X_test_badnet.pkl')

    Y_train_badnet = load_label_pickle(root_path, pickle_name='Y_train_badnet.pkl')
    Y_test_badnet = load_label_pickle(root_path, pickle_name='Y_test_badnet.pkl')
    Y_test_benign = load_label_pickle(benign_path, pickle_name='Y_test_badnet.pkl')

    return X_train_badnet, Y_train_badnet, X_test_badnet, Y_test_badnet, X_test_benign, Y_test_benign


# model evaluation
def evaluation(model,X_test, Y_test,mods, lbl, snrs, test_idx ):
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

def evaluation_avg(preds1,preds2, Y_test, lbl, snrs, test_idx ):
    # model evaluation
    acc = []
    for snr in snrs:
        # Extract classes @ SNR
        test_SNRs = list(map(lambda x: lbl[x][1], test_idx))
        test_idx_snr = np.where(np.array(test_SNRs) == snr)[0]

        # Weighted average of predictions
        # 假设你已经定义了权重 weight1 和 weight2
        weight1=0.6
        weight2=0.4
        preds_avg = (preds1[test_idx_snr] * weight1 + preds2[test_idx_snr] * weight2) / (weight1 + weight2)

        # 转换预测概率为最终预测的类别
        final_preds = np.argmax(preds_avg, axis=1)
        true_labels = np.argmax(Y_test[test_idx_snr], axis=1)

        # 计算准确率
        accuracy = np.mean(final_preds == true_labels)
        print(f"SNR: {snr}, Accuracy: {accuracy}")
        acc.append(accuracy)

    acc_mean = np.mean(acc)
    print(f"Overall Mean Accuracy: {acc_mean}")
    return acc_mean

def load_amr_model(pos_model_path):

    model = load_model(pos_model_path)

    print(model.summary())
    return model

def main():
    args = parse_arguments()
    POS_RATE = 0.1
    TRIGGER_TYPE = args.TRIGGER_TYPE

    # 加载seq数据
    X_train, X_test, Y_train, Y_test, mods, lbl, snrs, train_idx, test_idx = load_data()

    X_train_modified, Y_train_modified = set_trigger_config(X_train.copy(), Y_train.copy(), pos_rate=POS_RATE,
                                                            trigger_type=TRIGGER_TYPE, data_type='train')

    X_test_modified, Y_test_modified = set_trigger_config(X_test.copy(), Y_test.copy(), pos_rate=POS_RATE,
                                                            trigger_type=TRIGGER_TYPE, data_type='test')
    # 加载img数据
    # load img
    root_path = 'D:/zhaixu/Thesis_Code/datasets/constellation_' + TRIGGER_TYPE + '/'

    X_train_badnet, Y_train_badnet, X_test_badnet, Y_test_badnet, X_test_benign, Y_test_benign = load_img_datasets(
        root_path)

    model_seq_cnn = load_amr_model(pos_model_path="D:/zhaixu/Thesis_Code/CNN2_EPOCH_100.h5")
    model_seq_cldnn = load_amr_model(pos_model_path="D:/zhaixu/Thesis_Code/CLDNNLikeModel_EPOCH_100.h5")
    model_img = load_amr_model(pos_model_path="D:/zhaixu/Thesis_Code/VGG16_benign.h5")
    #model_feat = load_amr_model(pos_model_path="D:/zhaixu/Thesis_Code/VGG16_benign.h5")

# make prediction
    input_shape = model_seq_cnn.get_input_shape_at(0)
    X_test = X_test.reshape((X_test.shape[0],) + tuple(input_shape[1:]))
    pred_seq_cnn = model_seq_cnn.predict(X_test)
    evaluation(model_seq_cnn, X_test, Y_test, mods, lbl, snrs, test_idx)
    print(pred_seq_cnn.shape)

    input_shape = model_seq_cnn.get_input_shape_at(0)
    X_test = X_test.reshape((X_test.shape[0],) + tuple(input_shape[1:]))
    pred_seq_cldnn = model_seq_cldnn.predict(X_test)
    evaluation(model_seq_cldnn, X_test, Y_test, mods, lbl, snrs, test_idx)
    print(pred_seq_cldnn.shape)
# ensemble

    acc = evaluation_avg(pred_seq_cnn, pred_seq_cldnn, Y_test, lbl, snrs, test_idx)
    print(acc)

if __name__ == '__main__':
    main()