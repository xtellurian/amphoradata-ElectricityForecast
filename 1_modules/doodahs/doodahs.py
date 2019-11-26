from sklearn.ensemble import RandomForestRegressor
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def plot_two_vars(df_var_1,col_num1, df_var_2, col_num2, start='2019-11-10', end='2019-11-25', onediff=False, twodiff=False):
    """
    Plots two vars from pandas df cols within a certain range(start,end) of the index

    df_var_X: pd DataFrame containing df_var_X.index and df_var_X.iloc[:,col_numX]
    """
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
        df[name] = df.iloc[:,_]
        df[apple] = df.iloc[:,_].diff(_period)

    return df