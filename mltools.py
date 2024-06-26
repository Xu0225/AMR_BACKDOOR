import numpy as np
# import seaborn as sns
import pickle, random, sys
from tensorflow import keras
#import keras
from tensorflow.python.keras.utils import np_utils
# import keras.models as models
from tensorflow.keras import models
from tensorflow.python.keras.layers.core import Reshape, Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import BatchNormalization
from tensorflow.python.keras.layers.noise import GaussianNoise
from tensorflow.python.keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from tensorflow.python.keras.regularizers import *
from tensorflow.keras.callbacks import LearningRateScheduler
# from keras.optimizers import adam
from tensorflow.keras.optimizers import Adam

# from rmlmodel.Image.AlexNet import AlexNet
# from rmlmodel.Image.CNN import CNN
# from rmlmodel.Image.VGGlike import VGGLike
# from rmlmodel.Image.ZFNet import ZFNet

from rmlmodel.Sequence.vtcnn2 import VTCNN2
from rmlmodel.Sequence.CNN2 import CNN2
from rmlmodel.Sequence.CNN2Model import CNN2Model

from rmlmodel.Sequence.CLDNNLikeModel import CLDNNLikeModel
from rmlmodel.Sequence.CLDNNLikeModel1 import CLDNNLikeModel1
from rmlmodel.Sequence.CLDNNLikeModel2 import CLDNNLikeModel2

from rmlmodel.Sequence.CGDNN import CGDNN
from rmlmodel.Sequence.CuDNNLSTMModel import LSTMModel
#from rmlmodel.Sequence.DAE import DAE
from rmlmodel.Sequence.DCNNPF import DCNNPF
from rmlmodel.Sequence.DenseNet import DenseNet
from rmlmodel.Sequence.GRUModel import GRUModel
from rmlmodel.Sequence.ICAMC import ICAMC
from rmlmodel.Sequence.MCLDNN import MCLDNN
from rmlmodel.Sequence.MCNET import MCNET
from rmlmodel.Sequence.PETCGDNN import PETCGDNN
from rmlmodel.Sequence.ResNet import ResNet
from rmlmodel.Image.CNN import CNN
# 导入数据集

dbfile = open('D:/zhaixu/Thesis_Code/datasets/RML2016.10a_dict.dat', 'rb')
Xd = pickle.load(dbfile,encoding='latin1')

snrs,mods = map(lambda j: sorted(list(set(map(lambda x: x[j], Xd.keys())))), [1,0])
X = []
lbl = []
for mod in mods:
    for snr in snrs:
        X.append(Xd[(mod,snr)])
        for i in range(Xd[(mod,snr)].shape[0]):  lbl.append((mod,snr))
X = np.vstack(X)


# Partition the data
#  into training and test sets of the form we can train/test on
#  while keeping SNR and Mod labels handy for each
np.random.seed(2016)
n_examples = X.shape[0]
n_train = int(n_examples * 0.9)
train_idx = np.random.choice(range(0,n_examples), size=n_train, replace=False)
test_idx = list(set(range(0,n_examples))-set(train_idx))
X_train = X[train_idx]
X_test =  X[test_idx]
def to_onehot(yy):
    yy1 = np.zeros([len(yy), max(yy)+1])
    yy1[np.arange(len(yy)),yy] = 1
    return yy1
Y_train = to_onehot(list(map(lambda x: mods.index(lbl[x][0]), train_idx)))
Y_test = to_onehot(list(map(lambda x: mods.index(lbl[x][0]), test_idx)))

# print('数据集总数：',n_examples)
# print('调制方式' , len(mods),'种:' ,mods)
# print('信噪比:',snrs)


## load data

def calculate_amplitude_phase(X):
    amplitude = np.abs(X)
    phase = np.angle(X)
    return amplitude, phase

def get_fft_seq(IQ_sequence):
    # 将IQ序列转化为复数数组,提取FFT序列
    complex_sequence = IQ_sequence[0, :] + 1j * IQ_sequence[1, :]
    frequency_domain_representation = np.fft.fft(complex_sequence)

    # 分别获取实部和虚部
    real_part = np.real(frequency_domain_representation)
    imaginary_part = np.imag(frequency_domain_representation)

    # 将实部和虚部组合成两个维度为128的序列
    fft_seq = np.vstack((real_part, imaginary_part))

    return fft_seq


def get_fft_ap_seq(IQ_sequence):
    # 将IQ序列转化为复数数组,提取FFT序列
    complex_sequence = IQ_sequence[0, :] + 1j * IQ_sequence[1, :]
    frequency_domain_representation = np.fft.fft(complex_sequence)
    # 计算幅度相位
    amplitude, phase = calculate_amplitude_phase(frequency_domain_representation)

    fft_ap = np.vstack((amplitude, phase))
    return fft_ap

def FFT(signal):
    fs = 1  # 采样频率
    y = signal
    xf = np.fft.fft(y)  # 对离散数据y做fft变换得到变换之后的数据xf
    xfp = np.fft.fftfreq(len(y), d=1 / fs)  # fftfreq(length，fs)，计算得到频率
    xf = abs(xf)  # 将复数求模，得到fft的幅值

    # signal = np.stack((xf,xfp), axis=0)
    return xf


def trans_to_FFT(signal_set):
    for i in range(signal_set.shape[0]):
        # 三种生成方式
        # ① FFT(signal_set[i])：直接对IQ（2，128）进行fft
        # ② get_fft_seq(signal_set[i]): IQ转为复数形式，再做fft，保留complex信息
        # ② def get_fft_ap_seq(IQ_sequence): 提取②中的幅度、相位，构建频域AP序列，可选项
        #signal_set[i] = FFT(signal_set[i])
        signal_set[i] = get_fft_seq(signal_set[i])
        #signal_set[i] = get_fft_ap_seq(signal_set[i])
    return signal_set


# AP
def AP(signal):
    x = signal[1] / signal[0]
    X_p = np.arctan(x)
    X_a = (signal[0] ** 2 + signal[1] ** 2) ** 0.5
    signal = np.stack((X_a, X_p), axis=0)
    return signal


def trans_to_AP(signal_set):
    for i in range(signal_set.shape[0]):
        signal_set[i] = AP(signal_set[i])
    return signal_set


def get_seq_data(seq_data, seq_dtype="IQ"):
    if seq_dtype == "IQ":
        return seq_data
    elif seq_dtype == "AP":
        seq_data = trans_to_AP(seq_data)
        return seq_data
    elif seq_dtype == "FFT":
        seq_data = trans_to_FFT(seq_data)
        return seq_data


def fix_dim(X):
    if X.shape[1] == 2:
        X = X.swapaxes(2, 1)
    else:
        X = X
    return X


def build_model(target_model='CLDNN'):
    if target_model == 'VTCNN2':
        model = VTCNN2(weights=None, input_shape=[128, 2])

    elif target_model == 'CNN2Model':
        model = CNN2Model()

    elif target_model == 'CNN2':
        model = CNN2(None,input_shape=[2,128],classes=11)

    elif target_model == 'CLDNNLikeModel':
        model = CLDNNLikeModel()

    elif target_model == 'CLDNNLikeModel1':
        model = CLDNNLikeModel1()

    elif target_model == 'CLDNNLikeModel2':
        model = CLDNNLikeModel2()

    elif target_model == 'CGDNN':
        model = CGDNN()

    elif target_model == 'LSTMModel':
        model = LSTMModel()

    elif target_model == 'PETCGDNN':
        model = PETCGDNN()

    # elif target_model == 'DAE':
    #     model = DAE()

    elif target_model == 'DCNNPF':
        model = DCNNPF()

    elif target_model == 'DenseNet':
        model = DenseNet()

    elif target_model == 'GRUModel':
        model = GRUModel()

    elif target_model == 'ICAMC':
        model = ICAMC()

    elif target_model == 'MCLDNN':
        model = MCLDNN()

    elif target_model == 'MCNET':
        model = MCNET()

    elif target_model == 'PETCGDNN':
        model = PETCGDNN()

    elif target_model == 'ResNet':
        model = ResNet()

    elif target_model == 'CNN':
         model = CNN(input_shape=(75, 75, 3))
    # elif target_model == 'VGG':
    #     model = VGGLikeModel(weights=None, input_shape=[128, 2])

    # elif target_model == 'AlexNet':
    #     model = AlexNet(input_shape=(75, 75, 3))
    #
    # elif target_model == 'CNN':
    #     model = CNN(input_shape=(75, 75, 3))
    #
    # elif target_model == 'ZFNet':
    #     model = ZFNet(input_shape=(75, 75, 3))
    #
    # elif target_model == 'VGGLike':
    #     model = VGGLike(weights=None, input_shape=(75, 75, 3))

    adam = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer=adam)
    # model.summary()

    return model

# train model
# def scheduler(epoch):
#     print("epoch({}) lr is {}".format(epoch, K.get_value(model.optimizer.lr)))
#     return K.get_value(model.optimizer.lr)

def train(model, X_train, Y_train, X_val, Y_val, nb_epoch=100, batch_size=1024):
    #reduce_lr = LearningRateScheduler(scheduler)

    filepath = 'VGG_dr0.5.h5'
    history = model.fit(X_train,
                        Y_train,
                        batch_size=batch_size,
                        epochs=nb_epoch,
                        verbose=1,
                        validation_data=(X_val, Y_val),
                        callbacks=[

                                   keras.callbacks.ModelCheckpoint(filepath, monitor='val_loss', verbose=1,
                                                                   save_best_only=True, mode='auto'),
                                   keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, verbose=1,
                                                                     patince=5, min_lr=0.0000001),
                                   #keras.callbacks.EarlyStopping(monitor='val_loss', patience=50, verbose=1, mode='auto')
                                   ]
                        )
    return model, history


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

# model evaluation
def evaluation_key_snr(model,X_test, Y_test,test_idx,key_snr = 18):
    classes = mods
    acc = []
    for snr in snrs:
        if snr == key_snr:
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
            print("Overall Accuracy: ", cor / (cor+ncor))
            acc.append(1.0 * cor / (cor + ncor))
    return acc