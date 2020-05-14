"""
IN5410 - Energy informatics | Assignment 2

Data processing
"""
import os, random, sys, argparse, csv

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


def Data(TrainData, WF_input, Solution, meter=''):

    if meter == 'T1':
        # Fix data for the specific task
        TrainData.drop(columns=['U10', 'V10', 'U100', 'V100', 'WS100'], axis=1, inplace=True)
        WF_input.drop(columns =['U10', 'V10', 'U100', 'V100', 'WS100'], axis=1, inplace=True)

    elif meter == 'T2':
        TrainData.drop(columns=['U100', 'V100', 'WS100'], axis=1, inplace=True)
        WF_input.drop(columns =['U100', 'V100', 'WS100'], axis=1, inplace=True)

    elif meter == 'T3':
        TrainData.drop(columns=['U10','V10','WS10', 'U100', 'V100', 'WS100'], axis=1, inplace=True)

    else:
        print("Note: You are now using all data columns in dataset TrainData and WF_input")

    features = TrainData.loc[:, (TrainData.columns != 'POWER')].values
    target   = TrainData.loc[:, TrainData.columns == 'POWER'].values        # Targets

    pred_features  = WF_input.loc[:, WF_input.columns != 'POWER'].values    # Predicted power
    power_solution = Solution.loc[:, Solution.columns == 'POWER'].values    # Actual power data

    return features, target, pred_features, power_solution

# convert an array of values into a dataset matrix
def create_dataset(dataset, look_back=1):
	dataX, dataY = [], []

	for i in range(len(dataset)-look_back):
		a = dataset[i:(i+look_back), 0]
		dataX.append(a)
		dataY.append(dataset[i + look_back, 0])

	return np.array(dataX), np.array(dataY)

def Make_csv_dataset(prediction, time, name='test.csv'):

    df = pd.DataFrame({'Timestamp': time, 'Forecast prediction': prediction.flatten()})
    df.to_csv(name, encoding='utf-8', index=False)
