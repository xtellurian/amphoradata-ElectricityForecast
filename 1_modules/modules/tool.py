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

def plot_two_vars(df_var_1,col_num1, df_var_2, col_num2, start='2019-11-10', end='2019-11-25', onediff=False, twodiff=False):
    """
    Plots two vars from pandas df cols within a certain range(start,end) of the index

    df_var_X: pd DataFrame containing df_var_X.index and df_var_X.iloc[:,col_numX]
    """
    assert start!=end, 'Start < End criterion contradicted!'
    plt.xticks(rotation=45)

    col_name1 = df_var_1.iloc[:,col_num1].name
    if onediff:
        plt.plot(df_var_1[(df_var_1.index>=start) & (df_var_1.index <=end)].index,
        df_var_1[(df_var_1.index>=start) & (df_var_1.index <=end)].iloc[:,col_num1].diff(),
        label=col_name1)
    else:
        plt.plot(df_var_1[(df_var_1.index>=start) & (df_var_1.index <=end)].index,
        df_var_1[(df_var_1.index>=start) & (df_var_1.index <=end)].iloc[:,col_num1],
        label=col_name1)
    
    col_name2 = df_var_2.iloc[:,col_num2].name
    if twodiff:
        plt.plot(df_var_2[(df_var_2.index>=start) & (df_var_2.index <=end)].index,
        df_var_2[(df_var_2.index>=start) & (df_var_2.index <=end)].iloc[:,col_num2].diff(),
        label=col_name2)
    else:
        plt.plot(df_var_2[(df_var_2.index>=start) & (df_var_2.index <=end)].index,
        df_var_2[(df_var_2.index>=start) & (df_var_2.index <=end)].iloc[:,col_num2],
        label=col_name2)

    plt.legend(loc='best')

def create_diffs(df, cols, _period=1):
    """
    Creates differentiated columns, with a period of differentiation of _period index steps

    df: pd DataFrame
    cols: list of column numbers to be differentiated
    _period: (int) period for differentiation, may be negative

    return: same pd DataFrame plus new differentiated cols
    """

    for _ in cols:
        name = df.iloc[:,_].name
        apple = ''.join(['d_',name])
        df[apple] = df.iloc[:,_].diff(_period)

    return df

def prediction_intervalls(RFR, prediction_data, percentile=95):
    """
    Approximation of confidence intervals based on the U-statistic approach 
    detailed in https://arxiv.org/pdf/1404.6473.pdf

    RFR              randomforest model containing RFR.predict and RFR.estimators_
    prediction_data  pd.DataFrame column or pd.series containing the values the
                     predictions are to be made for
    percentile       the very thing
    """
    error_down = []
    error_up = []
    for _ in range(len(prediction_data)):
        predictions = []
        for pred in RFR.estimators_:
            predictions.append(pred.predict(prediction_data[_])[0])
        error_down.append(np.percentile(predictions, (100 - percentile) / 2. ))
        error_up.append(np.percentile(predictions, 100 - (100 - percentile) / 2.))
    
    return error_down, error_up

def import_n_statelabel(location):
    """
    Imports 2x *.csv into two pd dataframes, adds '_location' to the column names.

    location (str)

    returns: 2x pd.DataFrame 
    """
    cols_weather = ['temperature', 'rainProb', 'windSpeed',
       'windDirection', 'cloudCover',
       'pressure']
    cols_electr = ['price', 'scheduledGeneration']

    loc = []
    loc2 = []
    for _ in cols_weather:
        loc.append('_'.join([_,location]))
    for _ in cols_electr:
        loc2.append('_'.join([_,location]))
    weather_frame = pd.read_csv('2_raw_data/weather-{}.csv'.format(location),
                                infer_datetime_format=True,
                                index_col=0,
                                parse_dates=True,
                                dayfirst=True,
                                names=loc, 
                                skiprows=1)
    elek_frame= pd.read_csv('2_raw_data/electricity-{}.csv'.format(location),
                            infer_datetime_format=True,
                            index_col=0, 
                            parse_dates=True, 
                            dayfirst=True, 
                            names=loc2,
                            skiprows=1)
    
    return weather_frame, elek_frame

def linear_model_predictions(training_features,test_feautres,valid_param):
    """
    Optimises linear model based by reducing the OLS distance vetween 
    training_features and test_feautres. Creates predicitions based
    on valid_param.

    training_features: pd.DataFrame containg the 'x' data
    test_features: pd.DataFrame (or series) containing trhe 'y' data
    valid_param: data for which y values are to be inferred
    """
    model = linear_model.LinearRegression(n_jobs=-1)
    model.fit(training_features, test_feautres)
    return model, model.predict(valid_param)

def model_MSE(predictions, validation_data, print_result=False):
    mse = np.mean((predictions-validation_data)**2)
    if print_result:
        return print("The model is inaccurate by $%.2f on average." % mse[0])
    else:
        return mse[0]

def rel_error(model, predictions, test_data, print_result=False):
    rel = np.mean(100*np.abs((predictions-test_data)/test_data))
    if print_result:
        return print("The model has a relative error of {:.3}%.".format(rel[0]))
    else:
        return rel[0]

def split_dates_df(df_, _index=True, drop=True):
    """
    This functions takes a pd.DataFrame as argument, and splits the dates in the 
    index-column into day, month, year, week day.
    
    df_: pandas pd.DataFrame
    index: (bool/str) column to be treated is index. If index, pass True,
    else pass a string
    drop: (bool) delete old date column afterwards (only if index=False)
    """
    # Ensure we are operating on the index:
    if isinstance(_index,bool):
        # creating the new columns
        new_cols = ('year', 'month', 'day', 'dayofweek', 'dayofyear', 'is_quarter_end', 'hour')
        for _ in new_cols:
            # One new col per item in new_cols
            df_[_] = getattr(df_.index, _)
        
    # if _index != bool, e.g. _index='Banana'    
    else:
        # ensure it's a datetime, pandas often happy to import cols as wrong type:
        if not np.issubdtype(df_[_index].dtype, np.datetime64):
            df_[_index] = pd.to_datetime(df_[_index], infer_datetime_format = True)
            # creating the new columns
            new_cols = ('year','month','day','dayofweek','dayofyear','is_quarter_end', 'hour')
            for _ in new_cols:
                df_[_] = getattr(df_[_index],_)
        if drop:
            df_.drop(_index, axis=1, inplace=True)
    
    return df_

def time_shift(df, n, col=False, fill=np.nan):
    """
    To hard code the time-like nature of a time series dataset, this
    function shifts the the variable to be predicted by 1,2,3,....,n
    timesteps. Caveat: defined only for equal spaced timesteps.


    df: pd.DataFrame
    col: (bool/str) column name of the column to be shifted in time. 
    If boolean, shifts entire DataFrame by n timesteps
    n: (int) number pf timesteps/periods the column is shifted
    fill: (int/str/bool) value the empty cells are to be filled with
    """
    if isinstance(col,bool):
        return df.shift(-n, fill_value=fill)
    else:
        return df[col].shift(-n, fill_value=fill)

def reshape_for_CNN(df):
    """
    changes df from pd.DataFrame to np.array to reshape the
    array into the right shape of (nrow, ncols, 1) corresponding to
    (batch_size, steps, input_dim) necessary for Conv1D input.
    """
    
    arr = np.array(df.values)
    nrows, ncols = arr.shape # pylint: disable=unpacking-non-sequence
    
    return arr.reshape(nrows, ncols, 1)  # pylint: disable=too-many-function-args
   
def step_decay(epoch,initialLR=0.1,drop=0.5,drop_per_epoch=15):
    '''
    Sets the learning rate to gradually decay per epoch with a rate decay given by `rop`,
    and a drop step every `drop_per_epoch`.
    The changes every few apochs happen through the callbacks tensorflow submodule.
    c.f. https://www.tensorflow.org/api_docs/python/tf/keras/callbacks/LearningRateScheduler
    '''
    initial_learning_rate = initialLR
    drop = drop
    epochs_drop = drop_per_epoch
    learning_rate = initial_learning_rate * math.pow(drop,  
           math.floor((1+epoch)/epochs_drop))
    return LearningRateScheduler(learning_rate,verbose=0)

def MSE_CNN(predictions, true_values):
    """
    calculates the error per returned

    returns mse np.array, rel error np.array 
    """
    predictions = pd.DataFrame(predictions)
    predictions.columns = true_values.columns
    predictions.index = true_values.index
    
    mse = round(np.mean((predictions - true_values)**2),3)
    rel = round(np.mean(100*np.abs((predictions-true_values)/true_values)),3)
    
    return mse, rel

def MSE_CNN_2(predictions, true_values):
    """
    calculates the error per returned

    returns mse np.array, rel error np.array 
    """
    predictions = pd.DataFrame(predictions)
    predictions.index = true_values.index
    
    mse = round(np.mean((predictions - true_values)**2),3)
    rel = round(np.mean(100*np.abs((predictions-true_values)/true_values)),3)
    
    return mse, rel

def MSE_CNN_df(predictions, true_values):
    """
    calculates the error per returned

    returns mse np.array, rel error np.array 
    """
    mse = np.mean((predictions.reshape(len(predictions)) - np.array(true_values).reshape(len(true_values)))**2)
    
    return mse

def MSE_NN_3(predictions, true_values):
    """
    calculates the error per returned

    returns mse np.array, rel error np.array 
    """
    predictions = [predictions[_].reshape((len(predictions[_],))) for _ in range(len(predictions))]

    df = pd.DataFrame([*predictions]).T
    df.columns = true_values.columns
    df.index = true_values.index
    
    mse = round(np.mean((predictions - true_values)**2),3)
    rel = round(np.mean(100*np.abs((predictions-true_values)/true_values)),3)
    
    return mse, rel


def banana():
    '''
    easter egg.
    '''
    return print('banana')
