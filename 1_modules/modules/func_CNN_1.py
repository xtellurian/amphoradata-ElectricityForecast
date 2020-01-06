from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn import linear_model
import statsmodels.api as sm
import pandas as pd
import numpy as np
from datetime import date
# !conda install holidays
# import holidays
import matplotlib.pyplot as plt

from tensorflow.keras.callbacks import LearningRateScheduler # pylint: disable=import-error
import math
from tensorflow.keras.layers import Input, Dense, concatenate, LeakyReLU # pylint: disable=import-error
from tensorflow.keras.layers import Conv1D, Dropout, MaxPooling1D, Flatten # pylint: disable=import-error
from tensorflow.keras.models import Model # pylint: disable=import-error

def func_CNN_1(df_, nn_name='func_CNN'):
    """
    Creates a 1D CNN model with 4 separate inputs, one output (vector). To use, remember to reshape
    the input into the right (batch_size, steps, input_dim) shape!
    (c.f. e.g. https://github.com/bajibabu/GlottGAN/blob/master/keras/dcgan.py
    or https://datascience.stackexchange.com/questions/38957/keras-conv1d-for-simple-data-target-prediction,
    or )
    """
    # Create inpit layers:
    input_width = df_.shape[1]
    input_NSW = Input(shape=(input_width,1), name='NSW')
    input_QLD = Input(shape=(input_width,1), name='QLD')
    input_VIC = Input(shape=(input_width,1), name='VIC')
    input_SA = Input(shape=(input_width,1), name='SA')

    # create a dense, fully connected layer with a leaky relu activation fct:
    NSW = Conv1D(64, (5), padding='causal')(input_NSW)
    NSW = LeakyReLU(alpha=0.00)(NSW)
    QLD = Conv1D(64, (5), padding='causal')(input_QLD)
    QLD = LeakyReLU(alpha=0.00)(QLD)
    VIC = Conv1D(64, (5), padding='causal')(input_VIC)
    VIC = LeakyReLU(alpha=0.00)(VIC)
    SA = Conv1D(64, (5), padding='causal')(input_SA)
    SA = LeakyReLU(alpha=0.00)(SA)

    layer = concatenate([SA,VIC,QLD,NSW], axis=-1)

    conv_1 = Conv1D(32, (3), padding='causal', activation='relu')(layer)
    conv_1 = LeakyReLU(alpha=0.03)(conv_1)
    drop_1 = Dropout(0.4)(conv_1)
    flat = Flatten()(drop_1)
    Dense_1 = Dense(256, activation='relu')(flat)
    Dense_2 = Dense(64, activation='relu')(Dense_1)
    output_layer = Dense(4)(Dense_2)

    # model1.add(MaxPooling2D(pool_size=(2, 2)))
    # model1.add(Dropout(0.25))
    # model1.add(Flatten())

    model_CNN = Model(inputs = [input_NSW,input_QLD,input_VIC,input_SA], outputs=[output_layer], name=nn_name)

    return model_CNN

