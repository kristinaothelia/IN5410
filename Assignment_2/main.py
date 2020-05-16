"""
IN5410 - Energy informatics | Assignment 2
"""
#import os, random, xlsxwriter, sys, argparse, warnings, csv
import sys, argparse, warnings, csv

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
WF_input  = Data.Get_data(filename='/Data/WeatherForecastInput.csv')

timestamps = Solution.index
times_plot = pd.to_datetime(timestamps)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="ML for Wind Energy Forecasting")

    group = parser.add_mutually_exclusive_group()
    group.add_argument('-1', '--Task1', action="store_true", help="Task 1: Power forecasting using wind speed")
    group.add_argument('-2', '--Task2', action="store_true", help="Task 2: Power forecasting using wind speed & direction")
    group.add_argument('-3', '--Task3', action="store_true", help="Task 3: Power forecasting without weather data")

    # Optional argument for Task 1 and 3 ML methods
    parser.add_argument('-L', '--LR',    action='store_true', help="Linear Regression (OLS)", required=False)
    parser.add_argument('-K', '--KNN',   action='store_true', help="K-nearest neighbor", required=False)
    parser.add_argument('-S', '--SVR',   action='store_true', help="Support Vector Machine", required=False)
    parser.add_argument('-F', '--FFNN',  action='store_true', help="Feed Forward Neural Network", required=False)
    parser.add_argument('-R', '--RNN',  action='store_true',  help="Recurrent Neural Network", required=False)
    # Optional argument for plotting
    parser.add_argument('-X', '--Plot',  action='store_true', help="Plotting", required=False)
    # Optional argument for printing out possible warnings
    parser.add_argument('-W', '--Warnings', action='store_true', help="Warnings", required=False)

    if len(sys.argv) <= 1:
        sys.argv.append('--help')

    args  = parser.parse_args()

    Task1     = args.Task1
    Task2     = args.Task2
    Task3     = args.Task3
    Plot      = args.Plot
    LR        = args.LR
    KNN       = args.KNN
    SVR       = args.SVR
    FFNN      = args.FFNN
    RNN       = args.RNN
    Warnings  = args.Warnings

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
            Data.Make_csv_dataset(prediction=y_pred, time=timestamps, \
                                  name='Predictions/ForecastTemplate1-LR.csv')

            # Accuracy, R**2
            P.Metrics(power_solution, y_pred, method="Linear Regression (LR)", \
                      filename="Model_evaluation/Task1_RMSE_LR.txt")

            if Plot == True:     # Graphical illustration
                P.prediction_solution_plot(y_pred, power_solution, times_plot, \
                                           title="Linear Regression", \
                                           figname="Results/Task1_LR.png", savefig=True)


        elif KNN == True:

            print("k-Nearest Neighbors (kNN)\n")

            # k-Nearest Neighbors
            y_pred, power_solution, k, weights, p = UI.Task1_kNN_input(features, target, pred_features, power_solution)

            # Save predicted results in .cvs files
            Data.Make_csv_dataset(prediction=y_pred, time=timestamps, \
                                  name='Predictions/ForecastTemplate1-kNN.csv')

            # Accuracy metrics
            P.Metrics(power_solution, y_pred, param="k=%g, p=%g, weight=%s" %(k, p, weights), \
                      method="k-Nearest Neighbors (kNN)", filename="Model_evaluation/Task1_RMSE_kNN.txt")

            if Plot == True:    # Graphical illustration
                P.prediction_solution_plot(y_pred, power_solution, times_plot, \
                                           title="k-Nearest Neighbors (kNN). k=%g"%k, \
                                           figname="Results/Task1_kNN.png", savefig=True)

        elif SVR == True:

            print("Support Vector Regression (SVR)\n")

            # Support Vector Regression
            y_pred, power_solution, kernel, C, gamma, epsilon = UI.Task1_SVR_input(features, target, pred_features, power_solution)

            # Save predicted results in .cvs files
            Data.Make_csv_dataset(prediction=y_pred, time=timestamps, \
                                  name='Predictions/ForecastTemplate1-SVR.csv')

            # Accuracy metrics
            P.Metrics(power_solution, y_pred, param="kernel=%s, C=%g, gamma=%s, eps=%g" %(kernel, C, gamma, epsilon), \
                      method="Support Vector Regression (SVR)", filename="Model_evaluation/Task1_RMSE_SVR.txt")

            if Plot == True:    # Graphical illustration
                P.prediction_solution_plot(y_pred, power_solution, times_plot, \
                                           title="Support Vector Regression (SVR)", \
                                           figname="Results/Task1_SVR.png", savefig=True)

        elif FFNN == True:

            print("Feed Forward Neural Network (FFNN)\n")

            # Need to check if the best value is within the range, if not, adjust range
            #eta_vals   = [0.00001, 0.0001, 0.001, 0.01, 0.1]
            #lmbd_vals  = [0.00001, 0.0001, 0.001, 0.01, 0.1]
            eta_vals   = [0.0001, 0.001, 0.01, 0.1]
            lmbd_vals  = [0.0001, 0.001, 0.01, 0.1]

            # Legge inn en if eller noe her
            def Heatmap():
                # Calculating MSEs, RMSEs and R2s that to use in Heatmap_MSE_R2
                MSE_range, RMSE_range, R2_range \
                = ML.FFNN_Heatmap_MSE_R2(features, target, pred_features, power_solution, eta_vals, lmbd_vals, shuffle=False)

                # Creating heatmaps of MSE, RMSE and R2 to choose the best value
                P.Heatmap_MSE_R2(MSE_range, RMSE_range, R2_range, lmbd_vals, eta_vals,\
                                title='FFNN', figname='Model_evaluation/FFNN', savefigs=True)
            #Heatmap()

            # FFNN calculated with random hyperparameter, we must remember to use the best values
            y_pred, power_solution, activation, solver, alpha, learning_rate_init \
            = ML.FFNN(features, target, pred_features, power_solution, lmbd_vals, eta_vals, default=True, shuffle=False)

            #Save predicted results in .cvs files
            Data.Make_csv_dataset(prediction=y_pred, time=timestamps, \
                                  name='Predictions/ForecastTemplate1-NN.csv')

            # Accuracy
            P.Metrics(power_solution, y_pred, param="activation=%s, solver=%s, lambda=%f, eta=%f" %(activation, solver, alpha, learning_rate_init), \
                     method="Feed Forward Neural Network (FFNN)",\
                     filename="Model_evaluation/Task1_RMSE_FFNN.txt") # Prov %g istedenfor %f..?

            if Plot == True:    # Graphical illustration
                P.prediction_solution_plot(y_pred, power_solution, times_plot,\
                                      title="Feed Forward Neural Network (FFNN)",\
                                      figname="Results/Task1_FFNN.png", savefig=True)

        else:
            print("Pass an argument for ML method for Task 1 (-L, -K, -S, -F)")


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
        Data.Make_csv_dataset(prediction=y_pred_mlr, time=timestamps, \
                              name='Predictions/ForecastTemplate2.csv')

        # Accuracy
        P.Metrics_compare(power_solution, y_pred_lr, y_pred_mlr, \
                          filename="Model_evaluation/Task2_RMSE.txt")

        if Plot == True:    # Graphical illustration
            P.prediction_solution_plot_T2(y_pred_lr, y_pred_mlr, power_solution, times_plot, \
                                          title="LR and MLR", figname="Results/Task2.png", savefig=True)


    elif Task3 == True:

        print("Task 3:  LR, SVR, ANN, and RNN"); print("--"*20)
        """
        In some situations, we may not always have weather data, e.g.,
        wind speed data, at the windfarm location. In this task, we'll make
        wind power production forecasting with only windpower generation data.

        In the new training data file, we only have TIMESTAMP and POWER,
        which is called time-seriesdata. Apply LR, SVR, ANN, and RNN techniques
        to predict the wind power generation, for 11.2013.
        We then evaluate the prediction accuracy, by MSE, RMSE and R2, with
        the true wind power measurements (in the file Solution.csv).
        """

        # Remove U10, V10, WS10, U100, V100, WS100 from TrainData.csv
        # We will only use Target and Power solution now.
        features, target, pred_features, power_solution = Data.Data(TrainData, WF_input, Solution, meter='T3')

        print(power_solution.shape)

        # Kanskje bare droppe if paa denne? Det er liksom bare task 3 siden vi skal lage felles plott osv..
        # Men vi maa vel gjore GridSearch fortsatt.. Eller default fra task 1? (utenom RNN)
        if LR == True:

            print("LR and SVR\n")

            look_back = 1       # Hva burde denne vere..?

            trainX, trainY = Data.create_dataset(target, look_back)          # training data set
            testX, testY   = Data.create_dataset(power_solution, look_back)  # testing  data set

            y_pred_LR, y_pred_SVR, power_solution = ML.LR_SVR(trainX, trainY, testX, testY)


            P.Metrics(power_solution, y_pred_LR, param="", method="LR", filename="Model_evaluation/Task3_LR.txt")
            P.Metrics(power_solution, y_pred_SVR, param="", method="SVR", filename="Model_evaluation/Task3_SVR.txt")
            P.prediction_solution_plot_T3_1(y_pred_LR, y_pred_SVR, power_solution, times_plot[:-1], title="LR and SVR", figname='Results/Task3_LR_SVR.png', savefig=True)

            #Save predicted results in .cvs files
            Data.Make_csv_dataset(prediction=y_pred_LR, time=timestamps[:-1], \
                                  name='Predictions/ForecastTemplate3-LR.csv')
            Data.Make_csv_dataset(prediction=y_pred_SVR, time=timestamps[:-1], \
                                  name='Predictions/ForecastTemplate3-SVR.csv')

            P.Metrics_compare(power_solution, y_pred_LR, y_pred_SVR, filename="Model_evaluation/Task3_LR_SVR.txt", Task2=True)
            P.prediction_solution_plot_T3_1(y_pred_LR, y_pred_SVR, power_solution, times_plot[:-look_back], title="LR and SVR", figname='Results/Task3_LR_SVR.png', savefig=True)


            '''
            # some people are 'adding' the prediction (yhat) with testX  === yhat
            # and the testY with Xtest                                   === y
            # and then they calculate the rmse on the full/entire test set, maybe google/send mail about this
            yhat = y_pred_SVR #y_pred_LR
            # reshape testX from (719,1) to (719,)
            testX = testX.flatten()
            # invert scaling for forecast
            inv_yhat = np.concatenate((yhat, testX))
            # invert scaling for actual
            power_solution = power_solution.flatten()
            inv_y = np.concatenate((power_solution, testX))
            # calculate RMSE
            print(inv_y.shape, inv_yhat.shape)
            rmse = ML.RMSE(inv_y, inv_yhat)
            print(rmse)
            '''


        elif RNN == True:
            #https://machinelearningmastery.com/how-to-develop-a-skilful-time-series-forecasting-model/
            print("Recurrent Neural Network (RNN)\n")

            look_back = 3       # Skal nok ikke vere 1 her...?

            trainX, trainY = Data.create_dataset(target, look_back)
            testX, testY   = Data.create_dataset(power_solution, look_back)

            # Reshape input to be [samples, time steps, features]
            trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
            testX  = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))

            train_pred, test_pred = ML.RNN(look_back, trainX, trainY, testX, testY, summary=True)

            print(test_pred.shape); print(testY.shape)

            rmse = ML.RMSE(testY, test_pred); print(rmse)


            if Plot == True:    # Graphical illustration
                P.prediction_solution_plot_T3(test_pred, testY, \
                                              title="Recurrent Neural Network", \
                                              figname='Results/Task3_RNN.png', savefig=True)

            P.Metrics(testY, test_pred, param="", method="RNN", filename="Model_evaluation/Task3_RNN.txt")

            """
            # Denne delen maa inn naar hele koden fungerer! :)

            #Save predicted results in .cvs files
            Data.Make_csv_dataset(prediction=y_pred_FFNN, time=timestamps[:-1], \
                                  name='Predictions/ForecastTemplate3-FFNN.csv')
            Data.Make_csv_dataset(prediction=y_pred_RNN, time=timestamps[:-1], \
                                  name='Predictions/ForecastTemplate3-RNN.csv')

            # Accuracy
            P.Metrics_compare(power_solution, y_pred_FFNN, y_pred_RNN, filename="Model_evaluation/Task3_FFNN_RNN.txt")
            P.prediction_solution_plot_T3_1(y_pred_FFNN, y_pred_RNN, power_solution, times_plot[:-look_back], title="FFNN and RNN", figname='Results/Task3_FFNN_RNN.png', savefig=True)
            """

        else:
            print("Pass an argument for ML method for Task 3 (-R)")
