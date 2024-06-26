# -*- coding:utf-8 -*-
import argparse
import warnings

from sklearn.model_selection import train_test_split

warnings.filterwarnings('ignore')

import sys

sys.path.append('D:\\zhaixu\\Thesis_Code')


import numpy as np


from mltools import get_seq_data
from trigger_config import load_data
from trigger_config import set_trigger_config


def parse_arguments():
    parser = argparse.ArgumentParser(description='Process trigger configurations.')
    parser.add_argument('--TRIGGER_TYPE', type=str, default = 'hanning', help='Type of trigger (badnet, random_location, hanning, spectrum_shift, phase_shift, remapped_awgn)')
    parser.add_argument('--POS_RATE', type=float, default=0.1, help='Positive rate of samples to be injected with the trigger.')
    parser.add_argument('--VIEW', type=str, default='expert', help='view of feature')
    parser.add_argument('--MODEL_NAME', type=str, default='CART')

    return parser.parse_args()

def eval(model, X_test, Y_test, mods, lbl, snrs, test_idx):
    # model0:原模型测试
    classes = mods
    acc = []
    for snr in snrs:

        # extract classes @ SNR
        test_SNRs = list(map(lambda x: lbl[x][1], test_idx))

        # CA
    #     test_X_i = X_test_benign.values[np.where(np.array(test_SNRs)==snr)]
    #     test_Y_i = Y_test[np.where(np.array(test_SNRs)==snr)]

        #ASR
        test_X_i = X_test.values[np.where(np.array(test_SNRs)==snr)]
        test_Y_i = Y_test[np.where(np.array(test_SNRs)==snr)]

        test_Y_i_hat = model.predict(test_X_i)
        # estimate classes
        test_Y_i_hat = model.predict(test_X_i)
        conf = np.zeros([len(classes),len(classes)])
        confnorm = np.zeros([len(classes),len(classes)])
        for i in range(0,test_X_i.shape[0]):
            j = list(test_Y_i[i,:]).index(1)
            k = int(np.argmax(test_Y_i_hat[i,:]))
            conf[j,k] = conf[j,k] + 1
        for i in range(0,len(classes)):
            confnorm[i,:] = conf[i,:] / np.sum(conf[i,:])
        #plt.figure()
        #plot_confusion_matrix(confnorm, labels=classes, title="ConvNet Confusion Matrix (SNR=%d)"%(snr))

        cor = np.sum(np.diag(conf))
        ncor = np.sum(conf) - cor
        #print("Overall Accuracy: ", cor / (cor+ncor))
        print(cor / (cor+ncor))
        acc.append(1.0 * cor / (cor + ncor))
    acc_mean = sum(acc) / len(acc)
    print('acc_mean: ',acc_mean)

def train_model(X_train,Y_train,model_name = 'CART'):
    if model_name == 'CART':
        from sklearn import datasets, model_selection, metrics, tree, preprocessing
        # 导入决策树
        model = tree.DecisionTreeClassifier(max_depth=35)
        # 模型训练
        model.fit(X_train.values, Y_train)

        return model

    elif model_name == 'XGBoost':
        import xgboost as xgb
        from sklearn.multiclass import OneVsRestClassifier

        model_xgb = xgb.XGBClassifier(max_depth=20, tree_method='gpu_hist', gpu_id=0)
        clf_multilabel = OneVsRestClassifier(model_xgb)
        clf_multilabel.fit(X_train.values, Y_train)

        return clf_multilabel

    elif model_name == 'LightGBM':
        from lightgbm import LGBMClassifier
        from sklearn.multiclass import OneVsRestClassifier

        clf_multilabel = OneVsRestClassifier(LGBMClassifier(max_depth=10))
        clf_multilabel.fit(X_train.values, Y_train)

        return clf_multilabel

    elif model_name == 'CNN':
        import keras
        import keras.backend as K
        from keras.callbacks import LearningRateScheduler
        from keras.models import Sequential
        from keras.layers import Dense, Flatten, Conv1D, MaxPool1D, Dropout

        # build CNN model
        model = Sequential()
        model.add(Conv1D(64, 3, input_shape=(X_train.shape[1], 1), activation='relu'))  # convolution
        model.add(MaxPool1D(pool_size=2))  # pooling

        model.add(Conv1D(64, 3, activation='relu'))  # convolution
        model.add(MaxPool1D(pool_size=2))  # pooling

        model.add(Flatten())  # flatten
        model.add(Dense(128, activation='relu'))  # fc
        model.add(Dropout(0.5))  # dropout
        model.add(Dense(11, activation='softmax'))

        # model compile
        model.compile(loss=keras.losses.categorical_crossentropy,
                      optimizer=keras.optimizers.Adadelta(),
                      metrics=['accuracy'])
        model.summary()

        # model training
        # Set up some params
        nb_epoch = 1  # number of epochs to train on
        batch_size = 1024  # training batch size

        def scheduler(epoch):
            print("epoch({}) lr is {}".format(epoch, K.get_value(model.optimizer.lr)))
            return K.get_value(model.optimizer.lr)

        reduce_lr = LearningRateScheduler(scheduler)

        x_train, x_val, y_train, y_val = train_test_split(X_train, Y_train, test_size=0.111,
                                                          random_state=42)
        # reshape 2D to 3D ->  x_train.reshape(num_of_examples,num_of_features,num_of_signals)
        x_train = np.array(x_train).reshape(x_train.shape[0], x_train.shape[1], 1)
        x_val = np.array(x_val).reshape(x_val.shape[0], x_val.shape[1], 1)

        filepath = 'CNN_dr0.5.h5'
        history = model.fit(x_train,
                            y_train,
                            batch_size=batch_size,
                            epochs=nb_epoch,
                            verbose=1,
                            validation_data=(x_val, y_val),
                            callbacks=[reduce_lr,
                                       keras.callbacks.ModelCheckpoint(filepath, monitor='val_loss', verbose=1,
                                                                       save_best_only=True, mode='auto'),
                                       keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, verbose=1,
                                                                         patince=3, min_lr=0.000001),
                                       keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, verbose=1,
                                                                     mode='auto')
                                       ]
                            )

        return model

def data_processs(trigger_type = 'badnet',view = 'time'):
    X_train, X_test, Y_train, Y_test, mods, lbl, snrs, train_idx, test_idx = load_data()

    X_train_modified, Y_train_modified = set_trigger_config(X_train.copy(), Y_train.copy(), pos_rate=0.1,
                                                            trigger_type=trigger_type, data_type='train')

    # # REP = IQ/AP/FFT
    # X_train_modified = get_seq_data(X_train_modified, seq_dtype=seq_dtype)

    X_test_modified, Y_test_modified = set_trigger_config(X_test.copy(), Y_test.copy(), pos_rate=0.1,
                                                          trigger_type=trigger_type, data_type='test')

    # # REP = IQ/AP/FFT
    # X_test_modified = get_seq_data(X_test_modified, seq_dtype=seq_dtype)

    X_test_benign = X_test.copy()

    from utlits import form_features_time
    X_train_feature_modified = form_features_time(X_train_modified[:, :, :],view = view)
    X_test_feature_modified = form_features_time(X_test_modified[:, :, :],view = view)
    X_test_feature_benign = form_features_time(X_test_benign[:, :, :],view = view)
    # print("X_train_feature_modified", X_train_feature_modified.shape)
    # print("X_test_feature_modified", X_test_feature_modified.shape)
    # print("X_test_feature_benign", X_test_feature_benign.shape)

    if view == 'expert':
        from utlits import complex_accumulate_features
        # concat HOCs
        X_train_feature_modified = np.concatenate([X_train_feature_modified,complex_accumulate_features(X_train_modified)],axis = 1)
        X_test_feature_modified = np.concatenate([X_test_feature_modified,complex_accumulate_features(X_test_modified)],axis = 1)
        X_test_feature_benign = np.concatenate([X_test_feature_benign,complex_accumulate_features(X_test_benign)],axis = 1)

    from utlits import standardize_features
    X_train_modified, X_test_modified = standardize_features(X_train_feature_modified,X_test_feature_modified)
    X_train_modified, X_test_benign = standardize_features(X_train_feature_modified,X_test_feature_benign)


    # devide train and val data
    from sklearn.model_selection import train_test_split
    X_train_std, X_val_std, y_train, y_val = train_test_split(X_train_modified.copy(), Y_train.copy(), test_size=0.00001, random_state=42)
    # print("X_train_std,", X_train_std.shape)
    # print("X_val_std,", X_val_std.shape)
    # print("X_test_modified,", X_test_modified.shape)

    import pandas as pd
    X_train_std = pd.DataFrame(X_train_std)
    X_val_std = pd.DataFrame(X_val_std)

    X_train_modified = pd.DataFrame(X_train_modified)
    X_test_modified = pd.DataFrame(X_test_modified)
    X_test_benign = pd.DataFrame(X_test_benign)

    # inf,nan数据填充
    X_train_std = (X_train_std.replace([np.inf, -np.inf], np.nan)).fillna(value=0)
    X_val_std = (X_val_std.replace([np.inf, -np.inf], np.nan)).fillna(value=0)

    X_train_modified = (X_train_modified.replace([np.inf, -np.inf], np.nan)).fillna(value=0)
    X_test_modified = (X_test_modified.replace([np.inf, -np.inf], np.nan)).fillna(value=0)
    X_test_benign = (X_test_benign.replace([np.inf, -np.inf], np.nan)).fillna(value=0)

    return X_train_modified,Y_train_modified, \
           X_train_std,y_train,\
           X_test_benign,Y_test,\
           X_test_modified,Y_test_modified,\
           mods, lbl, snrs, train_idx, test_idx

if __name__ == '__main__':

    args = parse_arguments()

    import time

    st = time.time()

    X_train_modified, Y_train_modified, \
    X_train_std, y_train, \
    X_test_benign, Y_test, \
    X_test_modified, Y_test_modified, \
    mods, lbl, snrs, train_idx, test_idx = data_processs(trigger_type = args.TRIGGER_TYPE, view=args.VIEW)

    model = train_model(X_train_modified,Y_train_modified,model_name=args.MODEL_NAME)

    if args.MODEL_NAME == 'CNN':
        X_test_benign = np.array(X_test_benign).reshape(X_test_benign.shape[0], X_test_benign.shape[1], 1)
        X_test_modified = np.array(X_test_modified).reshape(X_test_modified.shape[0], X_test_modified.shape[1], 1)
    else:
        pass

    print('CA')
    eval(model, X_test_benign, Y_test, mods, lbl, snrs, test_idx)
    print('ASR')
    eval(model, X_test_modified, Y_test_modified, mods, lbl, snrs, test_idx)

    et = time.time()

    print('time cost',et - st,'s')