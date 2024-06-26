# -*- coding:utf-8 -*-
# ï¿½ï¿½ï¿½ï¿½ï¿½ï¿½ï¿½Ý¼ï¿½
import os
import pickle, random, sys
import time
import argparse

# »ñÈ¡ÏîÄ¿µÄ¸ùÄ¿Â¼
project_root = '/root/zx/Thesis_Code/'

# ½«ÏîÄ¿¸ùÄ¿Â¼Ìí¼Óµ½ sys.path ÖÐ
sys.path.append(project_root)

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from PIL import Image
from trigger_config import load_data,set_trigger_config
from mltools import get_seq_data

matplotlib.use('Agg')

def parse_arguments():
    parser = argparse.ArgumentParser(description='Process trigger configurations.')
    parser.add_argument('--TRIGGER_TYPE', type=str, default = 'badnet', help='Type of trigger (badnet, random_location, hanning, spectrum_shift, phase_shift, remapped_awgn)')
    parser.add_argument('--POS_RATE', type=float, default=0.1, help='Positive rate of samples to be injected with the trigger.')
    parser.add_argument('--REP', type=str, default='AP')

    return parser.parse_args()
    
def read_images_from_folder(folder_path, target_size=(75, 75)):
    image_data = []


    for filename in os.listdir(folder_path):
        if filename.endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(folder_path, filename)

            img = Image.open(image_path).resize(target_size)

            # img = img.convert("RGB")
            
            img_array = np.array(img)
            image_data.append(img_array)

            print(image_path)

    return image_data


def save_images_to_pickle(images, pickle_file):
    with open(pickle_file, 'wb') as f:
        pickle.dump(images, f)

def package_data(signal_data,label,lbl,train_idx,test_idx,root_path,signal_type = 'X_test',label_type = 'Y_test'):
    ## X_train
    folder_path = root_path + signal_type
    pickle_file = root_path + signal_type + '.pkl'
    pickle_file_label = root_path + label_type + '.pkl'

    # package image data
    images = read_images_from_folder(folder_path)
    save_images_to_pickle(images, pickle_file)
    save_images_to_pickle(label, pickle_file_label)
    
    # package lbl,train_idx,test_idx
    pickle_file_lbl = root_path + 'lbl.pkl'
    pickle_file_train_idx = root_path + 'train_idx.pkl'
    pickle_file_test_idx= root_path + 'test_idx.pkl'
    save_images_to_pickle(lbl, pickle_file_lbl)
    save_images_to_pickle(train_idx, pickle_file_train_idx)
    save_images_to_pickle(test_idx, pickle_file_test_idx)


def to_onehot(yy):
    yy1 = np.zeros([len(yy), max(yy) + 1])
    yy1[np.arange(len(yy)), yy] = 1
    return yy1


def plot_constellation(signal,index,signal_type = 'X_test/'):

    I = signal[0]
    Q = signal[1]

    # ï¿½ï¿½ï¿½ï¿½É¢ï¿½ï¿½Í¼
    plt.scatter(I, Q)
    plt.axis('off')
    plt.savefig(root_path + signal_type  + str(index) + '.jpg', dpi=200, bbox_inches='tight')
    plt.clf()



if __name__ == '__main__':

    args = parse_arguments()

    root_path = '/root/zx/Thesis_Code/datasets/constellation_'+ args.TRIGGER_TYPE + '/'
    os.makedirs(root_path+'X_train_badnet')
    os.makedirs(root_path+'X_test_badnet')

    # load_data
    X_train,X_test,Y_train,Y_test,mods,lbl,snrs,train_idx,test_idx = load_data()

    # trigger inject
    X_train_modified, Y_train_modified = set_trigger_config(X_train.copy(), Y_train.copy(), pos_rate=args.POS_RATE,
                                                            trigger_type=args.TRIGGER_TYPE, data_type='train')
    
    X_test_modified, Y_test_modified = set_trigger_config(X_test.copy(), Y_test.copy(), pos_rate=args.POS_RATE,
                                                            trigger_type=args.TRIGGER_TYPE, data_type='test')
    if args.REP == 'AP':
        X_train_modified = get_seq_data(X_train_modified, seq_dtype = args.REP)
        X_test_modified = get_seq_data(X_test_modified, seq_dtype = args.REP)
    
    # plot constellation
    ## plot X_train
    plt.figure(figsize=(5, 5))
    num = X_train.shape[0]
    #num = 5
    for i in range(num):
        plot_constellation(X_train_modified[i],i,signal_type  = 'X_train_badnet/')
        #print('X_train'+args.TRIGGER_TYPE + str(i))
    

    ## plot X_test
    num = X_test.shape[0]
    #num = 5
    for i in range(num):
        plot_constellation(X_test_modified[i],i,signal_type = 'X_test_badnet/')
        #print('X_test'+args.TRIGGER_TYPE + str(i))

    # package data
    package_data(X_train_modified, Y_train_modified, lbl, train_idx, test_idx,root_path, signal_type='X_train_badnet', label_type='Y_train_badnet')
    print('Done making X_train_'+ args.TRIGGER_TYPE)
    package_data(X_test_modified, Y_test_modified, lbl, train_idx, test_idx, root_path, signal_type='X_test_badnet',label_type='Y_test_badnet')
    print('Done making X_test_'+ args.TRIGGER_TYPE)




