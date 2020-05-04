"""
IN5410 - Energy informatics | Assignment 2

This file...

"""
import os, random, xlsxwriter, sys, argparse

import numpy               	as np
import pandas               as pd
import seaborn              as sns

from sklearn.preprocessing import StandardScaler
# -----------------------------------------------------------------------------


def Get_data(filename='/TrainData.csv'):
    """
    Function for reading csv files
    Input: Filename as a string
    https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.read_csv.html
    """
    cwd      = os.getcwd()
    fn       = cwd + filename
    nanDict  = {}
    Data     = pd.read_csv(fn, header=0, skiprows=0, index_col=0, na_values=nanDict)
    return Data


def Data(TrainData, WF_input, Solution):
    features = TrainData.loc[:, (TrainData.columns != 'POWER')].values
    target   = TrainData.loc[:, TrainData.columns == 'POWER'].values        # Targets

    pred_features  = WF_input.loc[:, WF_input.columns != 'POWER'].values    # Targets
    power_solution = Solution.loc[:, Solution.columns == 'POWER'].values    # Targets

    return features, target, pred_features, power_solution

# ----------------------------------------------------------------------------
# Maatte bare sette det inn i en def for det ble rart naar man kjorte main :P
def linreg_test():
    from sklearn.linear_model  import LinearRegression
    from sklearn.metrics       import mean_squared_error
    # Data for training the model (x_train and y_train?):
    TrainData = Get_data(filename='/TrainData.csv')

    # Data for later prediction (X_test?):
    WF_input  = Get_data(filename='/WeatherForecastInput.csv')

    # Solution used to calculate error of the predictions (y_test?)
    Solution  = Get_data(filename='/Solution.csv')

    # Does not seem like any of the datasets contain nan-values:

    #TrainData = TrainData.replace(r'^\s*$', np.nan, regex=True)
    #TrainData = pd.DataFrame.dropna(TrainData, axis=0, how='any')
    #WF_input  = WF_input.replace(r'^\s*$', np.nan, regex=True)
    #WF_input = pd.DataFrame.dropna(WF_input, axis=0, how='any')


    # 'Power' is the 'target' and the other columns are the 'features'
    # Only use the features for 10m, so dropping the 100m colums

    TrainData.drop(columns=['U100', 'V100', 'WS100'], axis=1, inplace=True)
    WF_input.drop(columns =['U100', 'V100', 'WS100'], axis=1, inplace=True)

    print(TrainData)

    features, target, pred_features, power_solution = Data(TrainData, WF_input, Solution)

    linreg = LinearRegression()
    linreg.fit(features, target) #training the algorithm

    y_pred = linreg.predict(pred_features)

    compare_values = pd.DataFrame({'Actual': power_solution.flatten(), 'Predicted': y_pred.flatten()})
    print(compare_values)

    mse = mean_squared_error(power_solution, y_pred)
    print("mse_linreg: ", mse)

    #squared:boolean value, optional (default = True)
    #If True returns MSE value, if False returns RMSE value.
    # Funker ikke...
    #rmse = mean_squared_error(power_solution, y_pred, squared=False)
    rmse = np.sqrt(mean_squared_error(power_solution, y_pred))
    print("rmse_linreg: ", rmse)

#linreg_test()

#sc_feature = StandardScaler()
#sc_target = StandardScaler()
#feature = sc_X.fit_transform(X)
#target = sc_y.fit_transform(y)

# Save feature and target as arrays as we did in fys-stk? hmm..
