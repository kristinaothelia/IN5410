"""
IN5410 - Energy informatics | Assignment 2
"""
import os, random, xlsxwriter, sys, argparse, warnings, csv

import matplotlib.pyplot 	as plt
import numpy               	as np
import pandas               as pd
import seaborn              as sns

import Data                 as Data
import MachineLearning      as ML
import plots                as P
import user_input           as UI

# Python 3.7.4
#------------------------------------------------------------------------------

TrainData = Data.Get_data(filename='/Data/TrainData.csv')
Solution  = Data.Get_data(filename='/Data/Solution.csv')
F_temp    = Data.Get_data(filename='/Data/ForecastTemplate.csv')
WF_input  = Data.Get_data(filename='/Data/WeatherForecastInput.csv')

timestamps = F_temp.index #Solution.index. Endra til F_temp for vi bruker ikke denne til noe..? Den inneholder bare timestamps
times = pd.to_datetime(timestamps)

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

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="ML for Wind Energy Forecasting")

    group = parser.add_mutually_exclusive_group()
    group.add_argument('-1', '--Task1', action="store_true", help="Task 1")
    group.add_argument('-2', '--Task2', action="store_true", help="Task 2")
    group.add_argument('-3', '--Task3', action="store_true", help="Task 3")

    # Optional argument for methods (Task 1) and plotting
    parser.add_argument('-X', '--Plot', action='store_true', help="Plotting", required=False)
    parser.add_argument('-L', '--LR',   action='store_true', help="LR", required=False)
    parser.add_argument('-K', '--KNN',  action='store_true', help="kNN", required=False)
    parser.add_argument('-S', '--SVR',  action='store_true', help="SVR", required=False)
    parser.add_argument('-A', '--ANN',  action='store_true', help="ANN", required=False)

    # Optional argument for printing out possible warnings
    parser.add_argument('-W', '--Warnings', action='store_true', help="Warnings", required=False)

    if len(sys.argv) <= 1:
        sys.argv.append('--help')

    args  = parser.parse_args()

    Task1    = args.Task1
    Task2    = args.Task2
    Task3    = args.Task3
    Plot     = args.Plot
    LR       = args.LR
    KNN      = args.KNN
    SVR      = args.SVR
    ANN      = args.ANN
    Warnings = args.Warnings

    if not Warnings:
        # If the argument -W / --warnings is provided,
        # any warnings will be printed in the terminal
        warnings.filterwarnings("ignore")


    if Task1 == True:

        print("Task 1: LR, kNN, SVR and ANN"); print("--"*20)
        """
        Find the windspeed for the whole month of 11.2013 in the file
        WeatherForecastInput.csv. For each training model and the wind speed
        data, we predict the wind power generation in 11.2013 and save
        the predicted results in files.
        We then evaluate the prediction accuracy, by MSE, RMSE and R2, with
        the true wind power measurements (in the file Solution.csv).
        """

        # For all Task 1 methods.
        features, target, pred_features, power_solution = Data.Data(TrainData, WF_input, Solution, meter='T1')

        if LR == True:

            print("Linear Regression (LR)\n")

            # Linear Regression
            y_pred, power_solution = ML.linreg(features, target, pred_features, power_solution)

            # Save predicted results in .cvs files
            Data.Make_csv_dataset(prediction=y_pred, time=times, name='Predictions/ForecastTemplate1-LR.csv')

            # Accuracy, R**2
            P.Metrics(power_solution, y_pred, method="Linear Regression (LR)", filename="Model_evaluation/Task1_RMSE_LR.txt")

            if Plot == True:     # Graphical illustration
                P.prediction_solution_plot(y_pred, power_solution, times, title="Linear Regression", figname="Plots/Task1_LR.png", savefig=True)


        elif KNN == True:

            print("k-Nearest Neighbors (kNN)\n")

            # k-Nearest Neighbors
            y_pred, power_solution, k, weights, p = UI.Task1_kNN_input(features, target, pred_features, power_solution)

            # Save predicted results in .cvs files
            Data.Make_csv_dataset(prediction=y_pred, time=times, name='Predictions/ForecastTemplate1-kNN.csv')

            # Accuracy metrics
            P.Metrics(power_solution, y_pred, param="k=%g, p=%g, weight=%s" %(k, p, weights), method="k-Nearest Neighbors (kNN)", filename="Model_evaluation/Task1_RMSE_kNN.txt")

            if Plot == True:    # Graphical illustration
                P.prediction_solution_plot(y_pred, power_solution, times, title="k-Nearest Neighbors (kNN). k=%g"%k, figname="Plots/Task1_kNN.png", savefig=True)

        elif SVR == True:

            print("Support Vector Regression (SVR)\n")

            # Support Vector Regression
            y_pred, power_solution, kernel, C, gamma, epsilon = UI.Task1_SVR_input(features, target, pred_features, power_solution)

            # Save predicted results in .cvs files
            Data.Make_csv_dataset(prediction=y_pred, time=times, name='Predictions/ForecastTemplate1-SVR.csv')

            # Accuracy metrics
            P.Metrics(power_solution, y_pred, param="kernel=%s, C=%g, gamma=%s, eps=%g" %(kernel, C, gamma, epsilon), method="Support Vector Regression (SVR)", filename="Model_evaluation/Task1_RMSE_SVR.txt")

            if Plot == True:    # Graphical illustration
                P.prediction_solution_plot(y_pred, power_solution, times, title="Support Vector Regression (SVR)", figname="Plots/Task1_SVR.png", savefig=True)

        elif ANN == True:

            print("Artificial Neural Network (ANN)\n")

            y_pred, power_solution = ML.ANN(features, target, pred_features, power_solution)

            # Save predicted results in .cvs files
            Data.Make_csv_dataset(prediction=y_pred, time=times, name='Predictions/ForecastTemplate1-NN.csv')

            # Accuracy
            P.Metrics(power_solution, y_pred, method="Artificial Neural Network (ANN)", filename="Model_evaluation/Task1_RMSE_ANN.txt")

            if Plot == True:    # Graphical illustration
                P.prediction_solution_plot(y_pred, power_solution, times, title="Artificial Neural Network (ANN)", figname="Plots/Task1_ANN.png", savefig=True)

        else:
            print("Pass an argument for ML method for Task 1 (-L, -K, -S, -A)")


    elif Task2 == True:

        print("Task 2: LR and MLR"); print("--"*20)
        """
        Wind power generation may also depend on wind direction, temperature,
        and pressure, as well as wind speed. In this task, we focus on the
        relationship between windpower generation and two weather parameters
        (i.e., wind speed and wind direction).
        Note the zonal component (U10) and the meridional component (V10) of
        the wind forecast in the file TrainData.csv. Wind direction can be
        calculated by U10 and V10. Then, build a MLR model between wind power
        generation and two weather parameters. Finally, predict the wind power
        production for the whole month 11.2013.
        We then evaluate the prediction accuracy, by MSE, RMSE and R2, with
        the true wind power measurements (in the file Solution.csv).
        """
        # For MLR, we need to drop U100, V100 and WS100
        features_mlr, target_mlr, pred_features_mlr, power_solution_mlr = Data.Data(TrainData, WF_input, Solution, meter='T2')
        # For LR, we also need to drop U10 and V10:
        TrainData.drop(columns=['U10', 'V10'], axis=1, inplace=True)
        WF_input.drop( columns=['U10', 'V10'], axis=1, inplace=True)
        features_lr, target_lr, pred_features_lr, power_solution_lr = Data.Data(TrainData, WF_input, Solution)

        y_pred_lr,  power_solution  = ML.linreg(features_lr, target_lr, pred_features_lr, power_solution_lr)
        y_pred_mlr, power_solution = ML.linreg(features_mlr, target_mlr, pred_features_mlr, power_solution_mlr)

        # Save predicted results in .cvs files
        Data.Make_csv_dataset(prediction=y_pred_mlr, time=times, name='Predictions/ForecastTemplate2.csv')

        # Accuracy
        P.Metrics_compare(power_solution, y_pred_lr, y_pred_mlr, filename="Model_evaluation/Task2_RMSE.txt")

        if Plot == True:    # Graphical illustration
            P.prediction_solution_plot_T2(y_pred_lr, y_pred_mlr, power_solution, times, title="LR and MLR", figname="Plots/Task2.png", savefig=True)


    elif Task3 == True:

        print("Task 3:  LR, SVR, ANN, and RNN"); print("--"*20)
        """
        In some situations, we may not always have weather data, e.g.,
        wind speed data, at the windfarm location. In this task, we'll make
        wind power production forecasting with only windpower generation data.

        In the new training data file, we only have TIMESTAMP and POWER,
        which is called time-seriesdata.
        Apply LR, SVR, ANN, and recurrent neural network (RNN) techniques to
        predict wind power generation, for 11.2013.
        We then evaluate the prediction accuracy, by MSE, RMSE and R2, with
        the true wind power measurements (in the file Solution.csv).
        """

        # Remove U10, V10, WS10, U100, V100, WS100 from TrainData.csv
        features, target, pred_features, power_solution = Data.Data(TrainData, WF_input, Solution, meter='T3')

        """
        #ForecastTemplate3-LR.csv, ForecastTemplate3-SVR.csv, ForecastTemplate3-ANN.csv, ForecastTemplate3-RNN.csv

        # Accuracy
        P.Metrics_compare(power_solution, y_pred_lr, y_pred_mlr, filename="Model_evaluation/Task3_RMSE.txt")

        if Plot == True:    # Graphical illustration
            P.prediction_solution_plot(y_pred, power_solution, times, title="???", figname="Plots/Task3.png", savefig=True)
        """
