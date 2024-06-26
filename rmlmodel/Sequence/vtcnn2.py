# # coding=gbk
import warnings

warnings.filterwarnings('ignore')
import numpy as np
from matplotlib import pyplot as plt
import tensorflow as tf

tf.compat.v1.disable_eager_execution()

import os, random

# os.environ["KERAS_BACKEND"] = "theano"

os.environ["KERAS_BACKEND"] = "tensorflow"
os.environ["CUDA_VISIBLE_DEVICE"] = '0'
# os.environ["THEANO_FLAGS"]  = "floatX=float32"
# os.environ["THEANO_FLAGS"]  = "device=cuda%d"%(1)

import numpy as np
# import cv2

# import seaborn as sns
import pickle, random, sys
from tensorflow import keras
from tensorflow.python.keras.utils import np_utils
# import keras.models as models
from tensorflow.keras import models
from tensorflow.python.keras.layers.core import Reshape, Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import BatchNormalization
from tensorflow.python.keras.layers.noise import GaussianNoise
from tensorflow.python.keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from tensorflow.python.keras.regularizers import *
# from keras.optimizers import adam
from tensorflow.keras.optimizers import Adam
# import theano as th
# import theano.tensor as T
import os

WEIGHTS_PATH = ('resnet_like_weights_tf_dim_ordering_tf_kernels.h5')
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Conv1D, MaxPool1D, ReLU, Dropout, Softmax, MaxPooling1D
from tensorflow.keras.layers import LSTM

from tensorflow.keras.callbacks import LearningRateScheduler
import tensorflow.keras.backend as K
from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.keras.regularizers import *
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import model_from_json
# import tensorflow as tf
import numpy as np
# import tensorflow.compat.v1 as tf
# tf.enable_eager_execution()

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Conv2D, MaxPool2D
from tensorflow.keras.layers import Dense, Flatten
# from keras.layers.convolutional import Conv1D,MaxPooling1D
import importlib, sys

importlib.reload(sys)

# from mpl_toolkits.mplot3d import Axes3D

from sklearn.preprocessing import StandardScaler

# import matplotlib.pyplot as plt # plotting
import numpy as np  # linear algebra
import os  # accessing directory structure
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)

import numpy as np
import pandas as pd
import glob
import os
# import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras import Model, layers
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras import layers

# set Keras data format as channels_first
K.set_image_data_format('channels_last')
print(K.image_data_format())


def VTCNN2(weights=None,
           in_shp=[128, 2],
           classes=11,
           **kwargs):
    if weights is not None and not (os.path.exists(weights)):
        raise ValueError('The `weights` argument should be either '
                         '`None` (random initialization), '
                         'or the path to the weights file to be loaded.')
    dr = 0.5  # dropout rate (%)
    model = models.Sequential()
    model.add(Reshape([1] + in_shp, input_shape=in_shp))
    # conv1
    model.add(ZeroPadding2D((0, 2)))
    model.add(Convolution2D(256, (2, 3), padding='same', activation="relu", kernel_initializer='glorot_uniform'))
    model.add(Dropout(dr))
    # conv2
    model.add(ZeroPadding2D((0, 2)))
    model.add(Convolution2D(80, (2, 3), padding="same", activation="relu", kernel_initializer='glorot_uniform'))
    model.add(Dropout(dr))

    model.add(Flatten())
    model.add(Dense(256, activation='relu', kernel_initializer='he_normal', name="dense1"))
    model.add(Dropout(dr))
    model.add(Dense(classes, kernel_initializer='he_normal', name="dense2"))
    model.add(Activation('softmax'))
    model.add(Reshape([classes]))
    # Load weights.
    if weights is not None:
        model.load_weights(weights)

    return model


model = VTCNN2(weights=None, input_shape=[128, 2])
model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')
# plot_model(model, to_file='model_VGG1D.png',show_shapes=True) # print model
model.summary()

















