# # coding=gbk
import warnings
warnings.filterwarnings('ignore')
import numpy as np
from matplotlib import pyplot as plt
import tensorflow as tf
tf.compat.v1.disable_eager_execution()

# # coding=gbk
# Import all the things we need ---
#   by setting env variables before Keras import you can set up which backend and which GPU it use

import os,random
#os.environ["KERAS_BACKEND"] = "theano"

os.environ["KERAS_BACKEND"] = "tensorflow"
os.environ["CUDA_VISIBLE_DEVICE"]  = '0'
#os.environ["THEANO_FLAGS"]  = "floatX=float32"
#os.environ["THEANO_FLAGS"]  = "device=cuda%d"%(1)

import numpy as np
#import cv2

#import seaborn as sns
import pickle, random, sys
from tensorflow import keras
from tensorflow.python.keras.utils import np_utils
#import keras.models as models
from tensorflow.keras import models
from tensorflow.python.keras.layers.core import Reshape,Dense,Dropout,Activation,Flatten
from tensorflow.keras.layers import BatchNormalization
from tensorflow.python.keras.layers.noise import GaussianNoise
from tensorflow.python.keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from tensorflow.python.keras.regularizers import *
#from keras.optimizers import adam
from tensorflow.keras.optimizers import Adam
#import theano as th
#import theano.tensor as T
import os
WEIGHTS_PATH = ('resnet_like_weights_tf_dim_ordering_tf_kernels.h5')
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input,Dense,Conv1D,MaxPool1D,ReLU,Dropout,Softmax
from tensorflow.keras.layers import LSTM

from tensorflow.keras.callbacks import LearningRateScheduler
import tensorflow.keras.backend as K
from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.keras.regularizers import *
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import model_from_json
#import tensorflow as tf
import numpy as np
#import tensorflow.compat.v1 as tf
#tf.enable_eager_execution()

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Conv2D, MaxPool2D
from tensorflow.keras.layers import Dense, Flatten
import importlib,sys

importlib.reload(sys)

#from mpl_toolkits.mplot3d import Axes3D

from sklearn.preprocessing import StandardScaler

#import matplotlib.pyplot as plt # plotting
import numpy as np # linear algebra
import os # accessing directory structure
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import numpy as np
import pandas as pd
import glob
import os
#import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras import Model,layers
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.optimizers import RMSprop


#set Keras data format as channels_first
K.set_image_data_format('channels_last')
print(K.image_data_format())

def ConvBNReluUnit(input,kernel_size = 8,index = 0):
    x = Conv1D(filters=64, kernel_size=kernel_size, padding='same', activation='relu',kernel_initializer='glorot_uniform',
               name='conv{}'.format(index + 1))(input)
    # x = BatchNormalization(name='conv{}-bn'.format(index + 1))(x)
    x = MaxPool1D(pool_size=2, strides=2, name='maxpool{}'.format(index + 1))(x)
    return x

def CLDNNLikeModel2(weights=None,
             input_shape=[128,2],
             classes=11,
             **kwargs):
    if weights is not None and not (os.path.exists(weights)):
        raise ValueError('The `weights` argument should be either '
                         '`None` (random initialization), '
                         'or the path to the weights file to be loaded.')

    dr = 0.5  # dropout rate (%)
    tap = 8
    input = Input(input_shape,name='input')
    x = input

    #Cnvolutional Block
    L = 4
    for i in range(L):
        x = ConvBNReluUnit(x,kernel_size = tap,index=i)

    #LSTM Unit
    # batch_size,64,2
    x = LSTM(units=50,return_sequences = True)(x)
    x = LSTM(units=50)(x)

    #DNN
    x = Dense(128,activation='selu',name='fc1')(x)
    x = Dropout(dr)(x)
    x = Dense(128, activation='selu', name='fc2')(x)
    x = Dropout(dr)(x)
    x = Dense(classes,activation='softmax',name='softmax')(x)

    model = Model(inputs = input,outputs = x)

    # Load weights.
    if weights is not None:
        model.load_weights(weights)

    return model

















