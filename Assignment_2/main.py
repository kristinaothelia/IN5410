"""
IN5410 - Energy informatics | Assignment 2
"""
import os, random, xlsxwriter, sys, argparse, warnings, csv

import matplotlib.pyplot 	as plt
import numpy               	as np
import pandas               as pd
import seaborn              as sns

from scipy.optimize 		import linprog
from random 				import seed

import readData             as Data
import MachineLearning      as ML
import plots                as P

# Python 3.7.4
#------------------------------------------------------------------------------

TrainData = Data.Get_data(filename='/Data/TrainData.csv')
Solution  = Data.Get_data(filename='/Data/Solution.csv')
F_temp    = Data.Get_data(filename='/Data/ForecastTemplate.csv')
WF_input  = Data.Get_data(filename='/Data/WeatherForecastInput.csv')

timestamps = Solution.index
times = pd.to_datetime(timestamps)

# Maa gjores i hver task?
#features, target, pred_features, power_solution = Data.Data(TrainData, WF_input, Solution)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="ML for Wind Energy Forecasting")

    group = parser.add_mutually_exclusive_group()
    group.add_argument('-1', '--Task1', action="store_true", help="Task 1")
    group.add_argument('-2', '--Task2', action="store_true", help="Task 2")
    group.add_argument('-3', '--Task3', action="store_true", help="Task 3")

    # Optional argument for plotting
    parser.add_argument('-X', '--Plot', action='store_true', help="Plotting", required=False)
    parser.add_argument('-L', '--linreg', action='store_true', help="Linreg", required=False)
    parser.add_argument('-K', '--KNN', action='store_true', help="kNN", required=False)
    parser.add_argument('-S', '--SVR', action='store_true', help="SVR", required=False)
    parser.add_argument('-A', '--ANN', action='store_true', help="ANN", required=False)
    #parser.add_argument('-P', '--Params', action='store_true', help="Find parameters", required=False)

    # Optional argument for printing out possible warnings
    parser.add_argument('-W', '--Warnings', action='store_true', help="Warnings", required=False)

    if len(sys.argv) <= 1:
        sys.argv.append('--help')

    args  = parser.parse_args()

    Task1    = args.Task1
    Task2    = args.Task2
    Task3    = args.Task3
    Plot     = args.Plot
    linreg   = args.linreg
    KNN      = args.KNN
    SVR      = args.SVR
    ANN      = args.ANN
    #Params   = args.Params
    Warnings = args.Warnings

    if not Warnings:
        # If the argument -W / --warnings is provided,
        # any warnings will be printed in the terminal
        warnings.filterwarnings("ignore")

    if Task1 == True:

        print("Task 1")
        print("--"*20)

        """
        X_train						| features
        X_test						| pred_features
        y_train						| target
        y_test						| power_solution

        Find the windspeed for the whole month of 11.2013 in the file
        WeatherForecastInput.csv. For each training model and the wind speed
        data, you predict the wind power generation in 11.2013 and save
        the predictedresults in files.

        Finally, you evaluate the prediction accuracy.
        You comparethe predicted wind power and the true wind power
        measurements (in the file Solution.csv).
        Please use the error metric RMSE to evaluate and compare the prediction
        accuracy among the machine learningapproaches.
        """

        '''
        #Fra noen andre, men kult!
        data = TrainData
        weather_forecast = WF_input

        data['windspeed']  = data['WS10']
        data['zonal']      = data['U10']
        data['meridional'] = data['V10']
        weather_forecast['windspeed']  = weather_forecast['WS10']
        weather_forecast['zonal']      = weather_forecast['U10']
        weather_forecast['meridional'] = weather_forecast['V10']
        cmap = sns.cubehelix_palette(start=1, light=1, as_cmap=True)
        ax=sns.kdeplot(data['zonal'], data['meridional'], cmap=cmap, shade=True, cut=5)
        ax=sns.kdeplot(weather_forecast['zonal'], weather_forecast['meridional'], cmap='Reds', shade=False, cut=5,shade_lowest=False)
        '''


        if linreg == True:

            print("Linear Regression\n")
            # Data preprocessing
            features, target, pred_features, power_solution = Data.Data(TrainData, WF_input, Solution, meter='ten')

            # Linear Regression
            y_pred, power_solution = ML.linreg(features, target, pred_features, power_solution)

            # Save predicted results in .cvs files
            Data.Make_csv_dataset(prediction=y_pred, time=times, name='Predictions/ForecastTemplate1-LR.csv')

            # Accuracy, R**2
            print("MSE:           %.3f"% ML.MSE(power_solution, y_pred))
            print("RMSE:          %.3f"% ML.RMSE(power_solution, y_pred))
            print("R2 (variance): %.3f"% ML.R2(power_solution, y_pred))

            if Plot == True:
                # Graphical illustration
                P.prediction_solution_plot(y_pred, power_solution, times, title="Linear Regression")

        elif KNN == True:

            print("kNN")
            # Data preprocessing
            features, target, pred_features, power_solution = Data.Data(TrainData, WF_input, Solution, meter='ten')

            # Find parameters
            # Faar bare den hoyeste k-verdien... overfitting?
            #ML.kNN_parameters(features, target, pred_features, power_solution)

            # k-Nearest Neighbor
            k = 12
            weights = 'uniform'
            y_pred, power_solution = ML.kNN(features, target, pred_features, power_solution, k, weights)

            # Save predicted results in .cvs files
            Data.Make_csv_dataset(prediction=y_pred, time=times, name='Predictions/ForecastTemplate1-kNN.csv')

            # Accuracy metrics
            print("\nMetrics when k=%g, weight=%s" %(k, weights))
            print("MSE:           %.3f"% ML.MSE(power_solution, y_pred))
            print("RMSE:          %.3f"% ML.RMSE(power_solution, y_pred))
            print("R2 (variance): %.3f"% ML.R2(power_solution, y_pred))

            if Plot == True:
                # Graphical illustration
                P.prediction_solution_plot(y_pred, power_solution, times, title="k-Nearest Neighbors (kNN). k=%g"%k)

        elif SVR == True:

            print("SVR\n")

            y_pred, power_solution = ML.SVR_func(TrainData, WF_input, Solution)

            # Save predicted results in .cvs files
            Data.Make_csv_dataset(prediction=y_pred, time=times, name='Predictions/ForecastTemplate1-SVR.csv')

            # Accuracy metrics
            print("MSE:           %.3f"% ML.MSE(power_solution, y_pred))
            print("RMSE:          %.3f"% ML.RMSE(power_solution, y_pred))
            print("R2 (variance): %.3f"% ML.R2(power_solution, y_pred))

            if Plot == True:
                # Graphical illustration
                P.prediction_solution_plot(y_pred, power_solution, times, title="Support Vector Regression (SVR)")

        elif ANN == True:

            print("ANN\n")

            #y_pred, power_solution = ML.

            # Save predicted results in .cvs files
            Data.Make_csv_dataset(prediction=y_pred, time=times, name='Predictions/ForecastTemplate1-NN.csv')

            # Accuracy
            print("MSE:           %.3f"% ML.MSE(power_solution, y_pred))
            print("RMSE:          %.3f"% ML.RMSE(power_solution, y_pred))
            print("R2 (variance): %.3f"% ML.R2(power_solution, y_pred))

            if Plot == True:
                # Graphical illustration
                P.prediction_solution_plot(y_pred, power_solution, times, title="Artificial Neural Network (ANN)")

        else:
            print("Pass an argument for ML method for Task 1 (-L, -K, -S, -A)")


    elif Task2 == True:

        print("Task 2")


    elif Task3 == True:

        print("Task 3")
