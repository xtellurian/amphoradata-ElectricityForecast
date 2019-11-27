from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn import linear_model
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


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
    Creates differenciated columns, with a period of differentiation of _period index steps

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

def linear_model_predictions(training_features,test_feautres,predict_param):
    """
    Optimises linear model based by reducing the OLS distance vetween 
    training_features and test_feautres. Creates predicitions based
    on predict_param.
    """
    model = linear_model.LinearRegression()
    model.fit(training_features, test_feautres)
    return model, model.predict(predict_param)

def model_MAE(model, predictions, test_data, print_result=False):
    mae = np.mean(np.abs(predictions-test_data))
    if print_result:
        return print("The model is inaccurate by $%.2f on average." % mae[0])
    else:
        return mae[0]

def rel_error(model, predictions, test_data, print_result=False):
    rel = np.mean(100*np.abs((predictions-test_data)/test_data))
    if print_result:
        return print("The model has a relative error of {:.3}%.".format(rel[0]))
    else:
        return rel[0]
