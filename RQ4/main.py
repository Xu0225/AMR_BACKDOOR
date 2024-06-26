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
from tensorflow.keras.layers import Input, Flatten
from tensorflow.keras.layers import Dense
from trigger_config import load_data
from trigger_config import set_trigger_config
from tensorflow.keras.models import load_model
from mltools import train

from tensorflow.keras.layers import Layer, Softmax,Dropout
import tensorflow.keras.backend as K

class SimpleAttention(Layer):
    def __init__(self, **kwargs):
        super(SimpleAttention, self).__init__(**kwargs)

    def build(self, input_shape):
        self.W = self.add_weight(name='attention_weight',
                                 shape=(input_shape[-1], input_shape[-1]),
                                 initializer='random_normal',
                                 trainable=True)
        super(SimpleAttention, self).build(input_shape)

    def call(self, inputs):
        # 计算注意力分数
        attention_scores = K.dot(inputs, self.W)
        attention_scores = Softmax(axis=-1)(attention_scores)

        # 加权和
        output = K.batch_dot(attention_scores, inputs, axes=[1, 1])
        return output

    def compute_output_shape(self, input_shape):
        return input_shape

def parse_arguments():
    parser = argparse.ArgumentParser(description='Process trigger configurations.')
    parser.add_argument('--TRIGGER_TYPE', type=str, default = 'badnet', help='Type of trigger (badnet, random_location, hanning, spectrum_shift, phase_shift, remapped_awgn)')
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
def evaluation(model,X_test, Y_test,mods, lbl, snrs, train_idx, test_idx ):
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

from tensorflow.keras.models import Model
def get_intermediate_output(model,input,layer_name):
    intermediate_layer_model = Model(inputs=model.input,
                                     outputs=model.get_layer(layer_name).output)

    # 使用新模型对input进行预测，获取中间层特征
    intermediate_output = intermediate_layer_model.predict(input)
    return intermediate_output

def get_intermediate_train_test(X_train,X_test,pos_model_path,layer_name):
    model = load_model(pos_model_path)

    print(model.summary())

    input_shape = model.get_input_shape_at(0)
    X_train = X_train.reshape((X_train.shape[0],) + tuple(input_shape[1:]))
    X_test = X_test.reshape((X_test.shape[0],) + tuple(input_shape[1:]))

    intermediate_train = get_intermediate_output(model, X_train, layer_name)
    intermediate_test = get_intermediate_output(model, X_test, layer_name)
    return intermediate_train, intermediate_test

def concate_feat(X_train_seq,X_test_seq,X_train_img,X_test_img):
    intermediate_output_seq_train, intermediate_output_seq_test = \
        get_intermediate_train_test(X_train_seq,X_test_seq,pos_model_path='D:\zhaixu\Thesis_Code\dl_amc_backdoor\multi_mode\meta_model\IQ_CNN_benign.h5',layer_name='flatten_2')

    intermediate_output_img_train, intermediate_output_img_test = \
        get_intermediate_train_test(X_train_img,X_test_img,pos_model_path='D:\zhaixu\Thesis_Code\dl_amc_backdoor\multi_mode\meta_model\CNN_benign.h5',layer_name='dense_5')

    concate_feature_train = np.concatenate((intermediate_output_seq_train, intermediate_output_img_train), axis=1)
    concate_feature_test = np.concatenate((intermediate_output_seq_test, intermediate_output_img_test), axis=1)

    return concate_feature_train,concate_feature_test

def build_model_with_attention(input_shape):
    # 定义模型输入
    inputs = Input(shape=input_shape)

    # 注意力层
    attention_output = SimpleAttention()(inputs)

    # Flatten层，以便将多维输出转换为一维，以便与全连接层相连
    flatten_output = Flatten()(attention_output)

    # 后续的全连接层，加入Dropout和L2正则化
    dense1 = Dense(512, activation='relu')(flatten_output)
    dropout1 = Dropout(0.5)(dense1)  # Dropout比例设置为0.5
    dense2 = Dense(256, activation='relu')(dropout1)
    dropout2 = Dropout(0.5)(dense2)  # 再次添加Dropout

    predictions = Dense(11, activation='softmax')(dropout2)  # 假设有10个输出类别

    # 构建和编译模型
    model = Model(inputs=inputs, outputs=predictions)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

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

    concate_feature_train, concate_feature_test = concate_feat(X_train_modified, X_test_modified, X_train_badnet, X_test_badnet)

    concate_feature_train = np.array(concate_feature_train)
    concate_feature_test = np.array(concate_feature_test)

    input_shape = concate_feature_train.shape[1:]  # 获取融合特征的形状

    # 构建带有注意力机制的模型
    model = build_model_with_attention(input_shape)

    x_train, x_val, y_train, y_val = train_test_split(concate_feature_train, Y_train, test_size=0.111, random_state=42)

    # 创建一个简单的全连接网络模型
    model = Sequential([
        Dense(512, activation='relu', input_shape=(concate_feature_train.shape[1],)),  # 确保输入层匹配拼接后的特征数量
        Dense(256, activation='relu'),
        Dense(11, activation='softmax')  # 假设有10个输出类别
    ])

    # 编译模型
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    model, history = train(model, x_train, y_train, x_val, y_val, nb_epoch=20, batch_size=1024)

    # ASR
    print('ASR')
    ASR = evaluation(model, concate_feature_test, Y_test_modified,mods, lbl, snrs, train_idx, test_idx )

    # CA
    TRIGGER_TYPE = 'benign'
    # 加载seq数据
    X_train, X_test, Y_train, Y_test, mods, lbl, snrs, train_idx, test_idx = load_data()

    # 加载img数据
    # load img
    root_path = 'D:/zhaixu/Thesis_Code/datasets/constellation_' + TRIGGER_TYPE + '/'

    X_train_badnet, Y_train_badnet, X_test_badnet, Y_test_badnet, X_test_benign, Y_test_benign = load_img_datasets(
        root_path)

    concate_feature_train, concate_feature_test = concate_feat(X_train_modified, X_test_modified, X_train_badnet, X_test_badnet)

    concate_feature_train = np.array(concate_feature_train)
    concate_feature_test = np.array(concate_feature_test)

    print('CA')
    CA = evaluation(model, concate_feature_test, Y_test, mods, lbl, snrs, train_idx, test_idx)

    return ASR, CA

if __name__ == '__main__':
    ASR, CA = main()
    print('ASR')
    print(ASR)
    print('CA')
    print(CA)