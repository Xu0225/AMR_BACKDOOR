
# 导入数据集
import pickle, random, sys
sys.path.append('D:\\zhaixu\\Thesis_Code')

import numpy as np
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

print('数据集总数：',n_examples)
print('调制方式' , len(mods),'种:' ,mods)
print('信噪比:',snrs)

# 数据划分
X_train = fix_dim(X_train)
X_test = fix_dim(X_test)
#X_train = X_train.swapaxes(2,1)
#X_test = X_test.swapaxes(2,1)
x_train, x_val, y_train, y_val = train_test_split(X_train, Y_train, test_size=0.111, random_state=42)

# train
from mltools import train
from mltools import evaluation
import pandas as pd

# model training
EPOCH = 100
model_TBD = ['VTCNN2',
             'CNN2',
             'CNN2Model',
             'CLDNNLikeModel',
             'CLDNNLikeModel1',
             'CLDNNLikeModel2',
             'CGDNN',
             'LSTMModel',
             'DAE',
             #'DCNNPF',
             'DenseNet',
             'GRUModel',
             'ICAMC',
             #'MCLDNN',
             'MCNET',
             #'PETCGDNN',
             'ResNet',
            ]
# model_TBD = ['DAE']
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

    # save_model
    root_path = 'D:/zhaixu/Thesis_Code/dl_amc_benign/'
    save_name = target_DNN + '_EPOCH_' + str(EPOCH) + '.h5'
    model.save(root_path + 'saved_model/' + save_name)

    # test_acc
    X_test = X_test.reshape((X_test.shape[0],) + tuple(input_shape[1:]))
    acc = evaluation(model, X_test, Y_test)

    # writing to Excel
    writer = pd.ExcelWriter(root_path + 'results/'  + target_DNN + '_EPOCH_' + str(EPOCH) + '.xlsx')
    df = pd.DataFrame(acc)

    df.to_excel(writer, 'acc', float_format='%.5f')

    writer.save()

    writer.close()
