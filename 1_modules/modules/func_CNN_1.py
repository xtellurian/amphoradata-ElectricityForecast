from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn import linear_model
#import statsmodels.api as sm
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

def func_CNN_1(df_, nn_name='func_CNN', single_output=False):
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
    all1 = Conv1D(128, (5), padding='causal')(input_all)
    all1 = LeakyReLU(alpha=0.01)(all1)
    all1 = MaxPooling1D(2)(all1)
    
    # create a cnn layer with a leaky relu activation fct:
    all2 = Conv1D(96, (5), padding='causal')(all1)
    all2 = LeakyReLU(alpha=0.01)(all2)
    all2 = MaxPooling1D(2)(all2)

    def cnn_lrelu_block(name_in, filters=32, kernel=3, padding='causal', lrelualpha=0.01, dropout=0.2, name=''):
        '''
        name_in being a tf tensor
        '''
        name = Conv1D(filters, (kernel), padding=padding, name=name)(name_in)
        name_out = LeakyReLU(alpha=lrelualpha)(name)
        name_out = Dropout(dropout)(name_out)
        return name_out
    
    # CNN block for all four states
    NSW1 = cnn_lrelu_block(all1, filters=32, kernel=3, name='NSW1')
    NSW1 = MaxPooling1D(2)(NSW1)
    QLD1 = cnn_lrelu_block(all1, filters=32, kernel=3, name='QLD1')
    QLD1 = MaxPooling1D(2)(QLD1)
    VIC1 = cnn_lrelu_block(all1, filters=32, kernel=3, name='VIC1')
    VIC1 = MaxPooling1D(2)(VIC1)
    SA1 = cnn_lrelu_block(all1, filters=32, kernel=3, name='SA1') 
    SA1 = MaxPooling1D(2)(SA1)

    # CNN block for all four states
    NSW2 = cnn_lrelu_block(NSW1, filters=32, kernel=3, name='NSW2')
    QLD2 = cnn_lrelu_block(QLD1, filters=32, kernel=3, name='QLD2')
    VIC2 = cnn_lrelu_block(VIC1, filters=32, kernel=3, name='VIC2')
    SA2 = cnn_lrelu_block(SA1, filters=32, kernel=3, name='SA2')
    
    # Dense block
    NSW2 = Flatten()(NSW2)
    QLD2 = Flatten()(QLD2)
    VIC2 = Flatten()(VIC2)
    SA2 = Flatten()(SA2)
    
    def dense_block(name_in,start,mid,end,dropout=0.2, name=''):
        Name2 = Dense(start)(name_in)
        name21 = LeakyReLU(alpha=0.03)(Name2)
        drop_3 = Dropout(dropout)(name21)
        Name4 = Dense(mid)(drop_3)
        name41 = LeakyReLU(alpha=0.03)(Name4)
        output_dense = Dense(end, name=name)(name41)
        return output_dense
    
    NSW3 = dense_block(NSW2,32,16,1, name='NSW')
    QLD3 = dense_block(QLD2,32,16,1, name='QLD')
    VIC3 = dense_block(VIC2,32,16,1, name='VIC')
    SA3 = dense_block(SA2,32,16,1, name='SA')
    
    # model1.add(MaxPooling2D(pool_size=(2, 2)))
    # model1.add(Dropout(0.25))
    # model1.add(Flatten())
    if single_output:
        layer = concatenate([NSW3,QLD3,VIC3,SA3], axis=-1)
        Dense_1 = Dense(32, activation='relu')(layer)
        output_layer = Dense(4)(Dense_1)
        model_CNN = Model(inputs = input_all, outputs=output_layer, name=nn_name)

    else:
        model_CNN = Model(inputs = input_all, outputs=[NSW3,QLD3,VIC3,SA3], name=nn_name)

    return model_CNN
