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

def func_NN_3(df_, nn_name='func_CNN'):
    """
    Creates a 1D dense nn model, with one input, four separate outputs. To use, remember to reshape the input into
    the right (batch_size, steps, input_dim) shape!
    (c.f. e.g. https://github.com/bajibabu/GlottGAN/blob/master/keras/dcgan.py
    or https://datascience.stackexchange.com/questions/38957/keras-conv1d-for-simple-data-target-prediction,
    or )
    """

    # Create inpit layers:
    input_width = df_.shape[1]
    input_all = Input(shape=(input_width,), name='all')

    # create a dense, fully connected layer with a leaky relu activation fct:
    all1 = Dense(1024, activation='relu')(input_all)
    
    def dense_block(name_in,start,mid,end,activation='relu',dropout=0.1, lrelualpha=0.03, name=''):
        Name2 = Dense(start, activation=activation)(name_in)
        drop_3 = Dropout(dropout)(Name2)
        Name4 = Dense(mid, activation=activation)(drop_3)
        output_dense = Dense(end)(Name4)
        name_out = LeakyReLU(alpha=lrelualpha, name=name)(output_dense)
        return name_out

    # Dense block for all four states
    NSW1 = dense_block(all1, 512,256,128, name='NSW1')
    NSW2 = dense_block(NSW1, 128,64,32, name='NSW2')
    QLD1 = dense_block(all1, 512,256,128, name='QLD1')
    QLD2 = dense_block(QLD1, 128,64,32, name='QLD2')
    VIC1 = dense_block(all1, 512,256,128, name='VIC1')
    VIC2 = dense_block(VIC1, 128,64,32, name='VIC2')
    SA1 = dense_block(all1, 512,256,128, name='SA1')
    SA2 = dense_block(SA1, 128,64,32, name='SA2')
    
    
    NSW3 = dense_block(NSW2,64,32,1, name='NSW')
    QLD3 = dense_block(QLD2,64,32,1, name='QLD')
    VIC3 = dense_block(VIC2,64,32,1, name='VIC')
    SA3 = dense_block(SA2,64,32,1, name='SA')
    
    # model1.add(MaxPooling2D(pool_size=(2, 2)))
    # model1.add(Dropout(0.25))
    # model1.add(Flatten())
    model_NN = Model(inputs = input_all, outputs=[NSW3,QLD3,VIC3,SA3], name=nn_name)

    return model_NN