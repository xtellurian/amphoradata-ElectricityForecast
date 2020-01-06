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

def func_CNN_2(df_, nn_name='func_CNN', single_output=False):
    """
    Creates a 1D CNN model, with one input, four separate outputs. To use, remember to reshape the input into
    the right (batch_size, steps, input_dim) shape!
    (c.f. e.g. https://github.com/bajibabu/GlottGAN/blob/master/keras/dcgan.py
    or https://datascience.stackexchange.com/questions/38957/keras-conv1d-for-simple-data-target-prediction,
    or )
    """
    # Create inpit layers:
    input_width = df_.shape[1]
    input_all = Input(shape=(input_width,1), name='all')

    # create a dense, fully connected layer with a leaky relu activation fct:
    all1 = Conv1D(256, (5), padding='causal')(input_all)
    all1 = LeakyReLU(alpha=0.03)(all1)

    def cnn_lrelu_block(name_in, filters=64, kernel=5, padding='causal', lrelualpha=0.03, name=''):
        '''
        name_in being a tf tensor
        '''
        name = Conv1D(filters, (kernel), padding=padding, name=name)(name_in)
        name_out = LeakyReLU(alpha=lrelualpha)(name)
        return name_out
    
    # CNN block for all four states
    NSW1 = cnn_lrelu_block(all1, name='NSW1')
    drop_NSW1 = Dropout(0.4)(NSW1)
    NSW2 = cnn_lrelu_block(drop_NSW1, filters=128, kernel=5, name='NSW2')
    QLD1 = cnn_lrelu_block(all1, name='QLD1')
    drop_QLD1 = Dropout(0.4)(QLD1)
    QLD2 = cnn_lrelu_block(drop_QLD1, filters=128, kernel=5, name='QLD2')
    VIC1 = cnn_lrelu_block(all1, name='VIC1')
    drop_VIC1 = Dropout(0.4)(VIC1)
    VIC2 = cnn_lrelu_block(drop_VIC1, filters=128, kernel=5, name='VIC2')
    SA1 = cnn_lrelu_block(all1, name='SA1')
    drop_SA1 = Dropout(0.4)(SA1)
    SA2 = cnn_lrelu_block(drop_SA1, filters=128, kernel=5, name='SA2')
    
    # Dense block
    NSW3 = Flatten()(NSW2)
    QLD3 = Flatten()(QLD2)
    VIC3 = Flatten()(VIC2)
    SA3 = Flatten()(SA2)
    
    def dense_block(name_in,start,mid,end,activation='relu',dropout=0.4, name=''):
        Name2 = Dense(start, activation=activation)(name_in)
        drop_3 = Dropout(dropout)(Name2)
        Name4 = Dense(mid, activation=activation)(drop_3)
        output_dense = Dense(end, activation=activation, name=name)(Name4)
        return output_dense
    
    NSW4 = dense_block(NSW3,512,64,1, name='NSW')
    QLD4 = dense_block(QLD3,512,64,1, name='QLD')
    VIC4 = dense_block(VIC3,512,64,1, name='VIC')
    SA4 = dense_block(SA3,512,64,1, name='SA')
    
    # model1.add(MaxPooling2D(pool_size=(2, 2)))
    # model1.add(Dropout(0.25))
    # model1.add(Flatten())
    if single_output:
        layer = concatenate([NSW4,QLD4,VIC4,SA4], axis=-1)
        Dense_1 = Dense(32, activation='relu')(layer)
        output_layer = Dense(4)(Dense_1)
        model_CNN = Model(inputs = input_all, outputs=output_layer, name=nn_name)

    else:
        model_CNN = Model(inputs = input_all, outputs=[NSW4,QLD4,VIC4,SA4], name=nn_name)

    return model_CNN