"""
IN5410 - Energy informatics | Assignment 2
"""
import os, random, xlsxwriter, sys, argparse, warnings

import matplotlib.pyplot 	as plt
import numpy               	as np
import pandas               as pd
import seaborn              as sns

from scipy.optimize 		import linprog
from random 				import seed

import readData             as Data
import MachineLearning      as ML

# Python 3.7.4
#------------------------------------------------------------------------------

TrainData = Data.Data(filename='/TrainData.csv')
Solution  = Data.Data(filename='/Solution.csv')
F_temp    = Data.Data(filename='/ForecastTemplate.csv')
WF_input  = Data.Data(filename='/WeatherForecastInput.csv')

print(TrainData)
print(Solution)
print(F_temp)
print(WF_input)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="ML for Wind Energy Forecasting")

    group = parser.add_mutually_exclusive_group()
    group.add_argument('-1', '--Task1', action="store_true", help="??")
    group.add_argument('-2', '--Task2', action="store_true", help="??")
    group.add_argument('-3', '--Task3', action="store_true", help="??")

    # Optional argument for plotting
    parser.add_argument('-X', '--Plot', action='store_true', help="Plotting", required=False)

    # Optional argument for printing out possible warnings
    parser.add_argument('-W', '--Warnings', action='store_true', help="Warnings", required=False)

    if len(sys.argv) <= 1:
        sys.argv.append('--help')

    args  = parser.parse_args()

    Task1    = args.Task1
    Task2    = args.Task2
    Task3    = args.Task3
    Plot     = args.Plot
    Warnings = args.Warnings

    if not Warnings:
        # If the argument -W / --warnings is provided,
        # any warnings will be printed in the terminal
        warnings.filterwarnings("ignore")

    if Task1 == True:

        print("Task 1")

        """
        Find the windspeed for the whole month of 11.2013 in the file
        WeatherForecastInput.csv. For each training modeland the wind speed
        data, you predict the wind power generation in 11.2013 and save
        the predictedresults in the files:
        ForecastTemplate1-LR.csv for the linreg model
        ForecastTemplate1-kNN.csv for the kNN model
        ForecastTemplate1-SVR.csv for the SVR model
        ForecastTemplate1-NN.csv for the neural networks model

        Finally, you evaluate the prediction accuracy.
        You comparethe predicted wind power and the true wind power
        measurements (in the file Solution.csv).
        Please usethe error metric RMSE to evaluate and compare the prediction
        accuracy among the machine learningapproaches.
        """


    elif Task2 == True:

        print("Task 2")


    elif Task3 == True:

        print("Task 3")