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

    # ��������в���
    parser.add_argument('--TRIGGER_TYPE', type=str, default='phase_shift')
    parser.add_argument('--GPU_NUM', type=str, default='0')
    parser.add_argument('--POS_RATE', type=float, default=0.1)
    parser.add_argument('--MODEL_NAME', type=str, default='CNN2')
    parser.add_argument('--EPOCH', type=int, default=2)

    return parser.parse_args()

def get_conv_index(model):
    # getting all indices where layer is convolutional layer
    # Ŀ�ģ���ȡģ�������о�����������
    # ԭ������ģ�͵����в㣬���ÿһ���Ƿ���Conv2D��Conv1D�㡣����ǣ����ò��������ӵ��б��С�
    convindex = []
    for i in range(len(model.layers)):
        layer = model.get_layer(index=i)
        if isinstance(layer, keras.layers.Conv2D) or isinstance(layer, keras.layers.Conv1D):
            convindex.append(i)
    return convindex

# # ����Ȩ�صײ� X �ٷֱȵĺ���
# def calc_bottom_X_percent_weight(weights, fraction):
#     # Ŀ�ģ���������ٷֱ�fraction��Ȩ����ֵ������ֵ����ȷ��Ȩ���޼��ĳ̶ȡ�
#     # ԭ������Ȩ�����飬�ҵ����ֵ����Сֵ��Ȼ�����fraction������ײ�X % ��Ȩ����ֵ��
#     # ��ʼ�����ֵ����СֵΪȨ�������ĵ�һ��Ԫ��
#     max = weights[0][0][0][0]
#     min = weights[0][0][0][0]
#
#     # ����Ȩ������������Ԫ�أ��ҵ����ֵ����Сֵ
#     for i in range(len(weights)):
#         for j in range(len(weights[i])):
#             for k in range(len(weights[i][j])):
#                 for m in range(len(weights[i][j][k])):
#                     if weights[i][j][k][m] < min:
#                         min = weights[i][j][k][m]
#                     if weights[i][j][k][m] > max:
#                         max = weights[i][j][k][m]
#
#     # ���ݸ����İٷֱȼ���ײ� X �ٷֱȵ�Ȩ��
#     truemin = min + (fraction * (max - min))
#
#     # ���ؼ���õ��ĵײ� X �ٷֱȵ�Ȩ��
#     return truemin



# # ��Ȩ��������С��ָ����ֵ��Ԫ������ĺ���
# def clear_min_weights(weights, thresh):
#     # ����Ȩ������������Ԫ��
#     for i in range(len(weights)):
#         for j in range(len(weights[i])):
#             for k in range(len(weights[i][j])):
#                 for m in range(len(weights[i][j][k])):
#                     # ���Ԫ��ֵС����ֵ��������Ϊ 0
#                     if weights[i][j][k][m] > thresh:
#                         weights[i][j][k][m] = 0
#
#     # ���ظ��º��Ȩ������
#     return weights

# ��Ȩ�������д���ָ����ֵ��Ԫ������ĺ���
def clear_max_weights(weights, thresh):
    # ����Ȩ������������Ԫ��
    for i in range(len(weights)):
        for j in range(len(weights[i])):
            for k in range(len(weights[i][j])):
                for m in range(len(weights[i][j][k])):
                    # ���Ԫ��ֵ������ֵ��������Ϊ 0
                    if weights[i][j][k][m] > thresh:
                        weights[i][j][k][m] = 0

    # ���ظ��º��Ȩ������
    return weights

# ����Ȩ�ض��� X �ٷֱȵĺ���
def calc_top_X_percent_weight(weights, fraction):
    # ��Ȩ������չƽ
    flat_weights = weights.flatten()
    # ������Ҫ������Ȩ������
    num_weights_to_keep = int(len(flat_weights) * (1 - fraction))
    # ��Ȩ�ؽ������򲢻�ȡ���� X% ����ֵ
    top_weights = np.sort(flat_weights)[-num_weights_to_keep]
    # ������ֵ
    return top_weights

# def fineprune(model, x):
#     # Ŀ�ģ���ģ���еľ�������Ȩ���޼���
#     # ���̣�
#     # �洢Ȩ�أ����ȣ���ȡ���о�����Ȩ�ء�
#     # ������ֵ������ÿ��������Ȩ�أ�����һ����СȨ����ֵ�������ֵ����ȷ����ЩȨ��Ӧ�ñ����㡣����ͨ��calc_bottom_X_percent_weight����ʵ�ֵģ�������Ȩ�صķ�Χ��һ�������İٷֱ�x�����㡣
#     # Ȩ���޼���ʹ��clear_min_weights������������С�ڼ��������ֵ��Ȩ����Ϊ0��
#     # ����ģ��Ȩ�أ���󣬸���ģ����ÿ��������Ȩ�غ�ƫ��
#     layer_weights = []
#     convindex = get_conv_index(model)
#     for i in convindex:
#         layer_weights.append(model.layers[i].get_weights()[0])
#
#     # ����ÿ����������СȨ����ֵ
#     min_weights_thr = []
#     for i in range(len(convindex)):
#         min_weights_thr.append(calc_bottom_X_percent_weight(layer_weights[i], x))
#
#     # ��ÿ����������Ȩ���޼�
#     new_weights = []
#     for i in range(len(convindex)):
#         new_weights.append(clear_min_weights(layer_weights[i], min_weights_thr[i]))
#
#     # �������������ӳ��
#     map_indices = {}
#     for i in range(len(convindex)):
#         map_indices[i] = convindex[i]
#
#     # Ϊ�˸���Ȩ�غ�ƫ�ã�����һ������Ȩ�غ�ƫ�õ��б�
#     weights_biases = [0 for x in range(2)]
#
#     # ����ģ�͵ľ����Ȩ��
#     for key in map_indices:
#         bias_weights = model.layers[map_indices[key]].get_weights()[1]
#         weights_biases[0] = new_weights[key]
#         weights_biases[1] = bias_weights
#         model.layers[map_indices[key]].set_weights(weights_biases)
#
#     return model

def fineprune(model, x):
    # �洢�����Ȩ��
    layer_weights = []
    convindex = get_conv_index(model)
    for i in convindex:
        layer_weights.append(model.layers[i].get_weights()[0])

    # ����ÿ�����������Ȩ����ֵ
    max_weights_thr = []
    for i in range(len(convindex)):
        max_weights_thr.append(calc_top_X_percent_weight(layer_weights[i], x))

    # ��ÿ����������Ȩ���޼�
    new_weights = []
    for i in range(len(convindex)):
        new_weights.append(clear_max_weights(layer_weights[i], max_weights_thr[i]))

    # �������������ӳ��
    map_indices = {}
    for i in range(len(convindex)):
        map_indices[i] = convindex[i]

    # Ϊ�˸���Ȩ�غ�ƫ�ã�����һ������Ȩ�غ�ƫ�õ��б�
    weights_biases = [0 for _ in range(2)]

    # ����ģ�͵ľ����Ȩ��
    for key in map_indices:
        bias_weights = model.layers[map_indices[key]].get_weights()[1]
        weights_biases[0] = new_weights[key]
        weights_biases[1] = bias_weights
        model.layers[map_indices[key]].set_weights(weights_biases)

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
        print("snr:",snr,"acc:",cor / (cor + ncor))
        acc.append(1.0 * cor / (cor + ncor))
    acc_mean = sum(acc) / len(acc)
    print('acc_mean: ',acc_mean)
    acc.append(acc_mean)
    return acc


def fineprune_and_finetune(model, initial_prune_percent=0.1, total_prune_target=0.9, prune_step=0.1, initial_epochs=10,
                           step_epochs=5, performance_threshold=0.01):
    # �洢ԭʼģ������
    original_accuracy = evalcustommodel(model, eval_type="CA")
    current_accuracy = original_accuracy
    current_prune_percent = initial_prune_percent

    # ��ʼ����ʽ��֦
    while current_prune_percent <= total_prune_target and current_accuracy >= (
            original_accuracy - performance_threshold):
        # ��֦
        model = fineprune(model, current_prune_percent)

        # ΢��
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        model.fit(x_train_clean, y_train_clean, validation_data=(x_val_clean, y_val_clean), epochs=initial_epochs)

        # ����
        current_accuracy = evalcustommodel(model, eval_type="CA")
        print(f"Pruning {current_prune_percent * 100}% - Current Accuracy: {current_accuracy}")

        # ׼����һ�ּ�֦
        current_prune_percent += prune_step
        initial_epochs = step_epochs  # ���ٺ���΢���ĵ�������

    # ���ؼ�֦��΢�����ģ��
    return model



def main():
    # load data
    X_train, X_test, Y_train, Y_test, mods, lbl, snrs, train_idx, test_idx = load_data()

    X_train_modified, Y_train_modified = set_trigger_config(X_train.copy(), Y_train.copy(), pos_rate=0.1,
                                                            trigger_type="hanning", data_type='train')

    x_train, x_val, y_train, y_val = train_test_split(X_train_modified, Y_train_modified, test_size=0.111,
                                                      random_state=42)

    # load model
    from tensorflow.keras.models import load_model
    # poisoned_model = load_model('D:/zhaixu/Thesis_Code/dl_amc_backdoor/all_to_one/saved_model/posioned_'+ args.MODEL_NAME + '_Hanning_EPOCH_100_POS_RATE_0.1.h5')

    root_path = 'D:/zhaixu/Thesis_Code/dl_amc_backdoor/all_to_one/saved_model/'
    # pos_model_name = f"{args.MODEL_NAME}_{args.TRIGGER_TYPE}_{args.POS_RATE}_{args.EPOCH}"
    # poisoned_model = load_model(root_path + pos_model_name + '.h5')
    poisoned_model = load_model(root_path + 'CNN2_spectrum_shift_0.1_100.h5')
    poisoned_model.summary()

    # reshape train input
    input_shape = poisoned_model.get_input_shape_at(0)
    X_train = X_train.reshape((X_train.shape[0],) + tuple(input_shape[1:]))
    X_train_modified = X_train_modified.reshape((X_train_modified.shape[0],) + tuple(input_shape[1:]))
    x_train = x_train.reshape((x_train.shape[0],) + tuple(input_shape[1:]))
    x_val = x_val.reshape((x_val.shape[0],) + tuple(input_shape[1:]))

    from shutil import copyfile, move
    # Loading the new weights in a temp model
    copyfile(root_path + 'CNN2_hanning_0.1_100.h5', root_path + 'temp_bd_net.h5')
    model_BadNet_new = load_model(root_path + 'temp_bd_net.h5')

    # no defense
    print('No defense eval\n')
    X_test_benign = X_test.reshape((X_test.shape[0],) + tuple(input_shape[1:]))

    X_test_modified, Y_test_modified = set_trigger_config(X_test.copy(), Y_test.copy(), pos_rate=0.1,
                                                          trigger_type="hanning", data_type='test')

    X_test_modified = X_test_modified.reshape((X_test_modified.shape[0],) + tuple(input_shape[1:]))

    print('CA')
    acc_ca_no_defense = evaluation(poisoned_model, X_test_benign, Y_test, test_idx, mods, snrs, lbl)
    print('ASR')
    acc_asr_no_defense = evaluation(poisoned_model, X_test_modified, Y_test_modified, test_idx, mods, snrs, lbl)


    x_train_clean, x_val_clean, y_train_clean, y_val_clean = train_test_split(X_train, Y_train, test_size=0.111,
                                                                              random_state=42)

    deviation = 0.1  # CA��ʧ����
    pruning_percent = 0.9 # ��֦Ȩ��
    poison_target = 0.01 # ASR�½�Ŀ��ֵ
    EPOCH = 1
    clean_acc_plt = []
    poison_acc_plt = []

    def evalcustommodel(bd_model, eval_type='ASR'):
        if eval_type == 'ASR':
            print('ASR')
            acc_asr_no_defense = evaluation(poisoned_model, X_test_modified, Y_test_modified, test_idx, mods, snrs, lbl)
            acc_mean = acc_asr_no_defense[-1]
        elif eval_type == 'CA':
            print('CA')
            acc_ca_no_defense = evaluation(bd_model, X_test_benign, Y_test, test_idx, mods, snrs, lbl)
            acc_mean = acc_ca_no_defense[-1]
        return acc_mean

    # ���㲢��¼ԭʼģ���ڸɾ����������ϵ�׼ȷ��
    acc_test_BadNetFP = evalcustommodel(model_BadNet_new, eval_type="CA")

    # ���㲢��¼ԭʼģ���ں��ж��������������ϵ�׼ȷ��
    acc_poison_BadNetFP = evalcustommodel(model_BadNet_new, eval_type="ASR")

    clean_acc_plt.append(acc_test_BadNetFP)
    poison_acc_plt.append(acc_poison_BadNetFP)

    # ����ƫ��ֵ����һ��׼ȷ�ʵ���ֵ����Ϊֹͣ�޼�������֮һ
    acc_cutoff = acc_test_BadNetFP - deviation
    step_accuracy = acc_cutoff
    print('Clean Accuracy cutoff', acc_cutoff)
    print("")

    count = 1

    while (step_accuracy >= acc_cutoff) and (acc_poison_BadNetFP >= poison_target):
        # ���� fineprune ��������ģ�ͽ����޼�
        model_BadNet_new = fineprune(model_BadNet_new, pruning_percent)

        # Fine tune
        model_BadNet_new.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

        history = model_BadNet_new.fit(x_train_clean, y_train_clean, validation_data=(x_val_clean, y_val_clean),
                                       epochs=EPOCH)

        # �����µĸɾ�����׼ȷ�ʺͶԿ�����׼ȷ��
        step_accuracy = evalcustommodel(model_BadNet_new, eval_type="CA")
        acc_poison_BadNetFP = evalcustommodel(model_BadNet_new, eval_type="ASR")

        # ��׼ȷ�ʴ洢����Ӧ���б���
        clean_acc_plt.append(step_accuracy)
        poison_acc_plt.append(acc_poison_BadNetFP)

        # �����ǰѭ����׼ȷ����Ϣ
        print('Clean accuracy:', step_accuracy)
        print("Poison accuracy:" + str(acc_poison_BadNetFP))
        print("")

        # ����ѭ��������
        count += 1

    # ѭ�������󣬱����޼����ģ�͵��ļ� "models/FP_GoodNet.h5"
    model_BadNet_new.save(root_path + "FP_GoodNet.h5")

    x_axis = np.arange(count)
    plt.plot(x_axis * 5, clean_acc_plt)
    plt.plot(x_axis * 5, poison_acc_plt)
    plt.legend(['Clean Test Accuracy', 'Poison Accuracy'])
    plt.xlabel("Pruned Channels Percent")
    plt.ylabel("Percent")
    plt.title("Clean and Poison Accuracies for Test dataset")

    import pandas as pd

    result_df = pd.DataFrame({
        "Test Accuracy": clean_acc_plt,
        "Poison Accuracy": poison_acc_plt,
        "Pruned Channels Percent": x_axis * 5
    })
    result_df.set_index("Pruned Channels Percent")

    print(result_df)

if __name__ == '__main__':
    import time
    start_time = time.time()

    main()

    end_time = time.time()
    print('time spend: ',end_time - start_time,'s')