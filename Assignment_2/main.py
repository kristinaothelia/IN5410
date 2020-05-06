"""
IN5410 - Energy informatics | Assignment 2
"""
import os, random, xlsxwriter, sys, argparse, warnings, csv

import matplotlib.pyplot 	as plt
import numpy               	as np
import pandas               as pd
import seaborn              as sns

#from scipy.optimize 		import linprog
#from random 				import seed

import readData             as Data
import MachineLearning      as ML
import plots                as P

# Python 3.7.4
#------------------------------------------------------------------------------

TrainData = Data.Get_data(filename='/Data/TrainData.csv')
Solution  = Data.Get_data(filename='/Data/Solution.csv')
F_temp    = Data.Get_data(filename='/Data/ForecastTemplate.csv')
WF_input  = Data.Get_data(filename='/Data/WeatherForecastInput.csv')

timestamps = F_temp.index #Solution.index. Endra til F_temp for vi bruker ikke denne til noe..? Den inneholder bare timestamps
times = pd.to_datetime(timestamps)


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

        # Ha denne her? Er lik for alle Task 1 oppgaver

        # For all Task 1 methods.
        features, target, pred_features, power_solution = Data.Data(TrainData, WF_input, Solution, meter='T1')

        if linreg == True:

            print("Linear Regression\n")
            # Data preprocessing
            #features, target, pred_features, power_solution = Data.Data(TrainData, WF_input, Solution, meter='T1')

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
                P.prediction_solution_plot(y_pred, power_solution, times, title="Linear Regression", figname="Plots/Task1_LR.png", savefig=True)


        elif KNN == True:

            print("kNN")
            # Data preprocessing
            #features, target, pred_features, power_solution = Data.Data(TrainData, WF_input, Solution, meter='T1')

            input = int(input("Do you want to: \n1) Use predefined values based on default- and GridSearchCV values? (Time efficient option) \n2) Use the 'best parameters' from GridSearchCV? \n3) Enter your own parameters based on a n_neighbors (k) plot? \nEnter 1, 2 or 3 (int): "))

            if input == 1:
                y_pred, power_solution, k, weights, p = ML.kNN(features, target, pred_features, power_solution, default=True)

            elif input == 2:
                best_params = ML.kNN_gridsearch(features, target, pred_features, power_solution)
                y_pred, power_solution, k, weights, p = ML.kNN(features, target, pred_features, power_solution, best_params, BestParams=True)

            elif input == 3:
                best_params = ML.kNN_gridsearch(features, target, pred_features, power_solution, plot=True)
                y_pred, power_solution, k, weights, p = ML.kNN(features, target, pred_features, power_solution, best_params, BestParams=False)

            else:
                print("Enter 1, 2 or 3, then enter"); exit()

            # Save predicted results in .cvs files
            Data.Make_csv_dataset(prediction=y_pred, time=times, name='Predictions/ForecastTemplate1-kNN.csv')

            # Accuracy metrics
            print("\nMetrics when k=%g, p=%g, weight=%s" %(k, p, weights))
            print("MSE:           %.3f"% ML.MSE(power_solution, y_pred))
            print("RMSE:          %.3f"% ML.RMSE(power_solution, y_pred))
            print("R2 (variance): %.3f"% ML.R2(power_solution, y_pred))

            if Plot == True:
                # Graphical illustration
                P.prediction_solution_plot(y_pred, power_solution, times, title="k-Nearest Neighbors (kNN). k=%g"%k, figname="Plots/Task1_kNN.png", savefig=True)

        elif SVR == True:

            print("SVR\n")

            input = int(input("Do you want to: \n1) Use predefined values based on default- and GridSearchCV values? (Time efficient option) \n2) Use the 'best parameters' from GridSearchCV? \nEnter 1 or 2 (int): "))

            if input == 1:
                y_pred, power_solution = ML.SVR_func(features, target, pred_features, power_solution)

            elif input == 2:
                y_pred, power_solution = ML.SVR_func(features, target, pred_features, power_solution, default=False)

            else:
                print("Enter 1 or 2, then enter"); exit()


            # Save predicted results in .cvs files
            Data.Make_csv_dataset(prediction=y_pred, time=times, name='Predictions/ForecastTemplate1-SVR.csv')

            # Accuracy metrics
            print("MSE:           %.3f"% ML.MSE(power_solution, y_pred))
            print("RMSE:          %.3f"% ML.RMSE(power_solution, y_pred))
            print("R2 (variance): %.3f"% ML.R2(power_solution, y_pred))

            if Plot == True:
                # Graphical illustration
                P.prediction_solution_plot(y_pred, power_solution, times, title="Support Vector Regression (SVR)", figname="Plots/Task1_SVR.png", savefig=True)

        elif ANN == True:

            print("ANN\n")

            #features, target, pred_features, power_solution = Data.Data(TrainData, WF_input, Solution, meter='T1')

            y_pred, power_solution = ML.ANN(features, target, pred_features, power_solution)

            # Save predicted results in .cvs files
            Data.Make_csv_dataset(prediction=y_pred, time=times, name='Predictions/ForecastTemplate1-NN.csv')

            # Accuracy
            print("MSE:           %.3f"% ML.MSE(power_solution, y_pred))
            print("RMSE:          %.3f"% ML.RMSE(power_solution, y_pred))
            print("R2 (variance): %.3f"% ML.R2(power_solution, y_pred))

            if Plot == True:
                # Graphical illustration
                P.prediction_solution_plot(y_pred, power_solution, times, title="Artificial Neural Network (ANN)", figname="Plots/Task1_ANN.png", savefig=True)

        else:
            print("Pass an argument for ML method for Task 1 (-L, -K, -S, -A)")


    elif Task2 == True:

        print("Task 2")

        features_mlr, target_mlr, pred_features_mlr, power_solution_mlr = Data.Data(TrainData, WF_input, Solution, meter='T2')
        # For LR:
        TrainData.drop(columns=['U10', 'V10'], axis=1, inplace=True)
        WF_input.drop(columns=['U10', 'V10'], axis=1, inplace=True)
        features_lr, target_lr, pred_features_lr, power_solution_lr = Data.Data(TrainData, WF_input, Solution)

        y_pred_lr,  power_solution  = ML.linreg(features_lr, target_lr, pred_features_lr, power_solution_lr)
        y_pred_mlr, power_solution = ML.linreg(features_mlr, target_mlr, pred_features_mlr, power_solution_mlr)

        # Save predicted results in .cvs files
        Data.Make_csv_dataset(prediction=y_pred_mlr, time=times, name='Predictions/ForecastTemplate2.csv')

        # Accuracy
        print("RMSE:          LR=%.3f, MLR=%.3f"% (ML.RMSE(power_solution, y_pred_lr), ML.RMSE(power_solution, y_pred_mlr)))
        print("R2 (variance): LR=%.3f, MLR=%.3f"% (ML.R2(power_solution, y_pred_lr), ML.R2(power_solution, y_pred_mlr)))

        if Plot == True:
            # Graphical illustration
            P.prediction_solution_plot_T2(y_pred_lr, y_pred_mlr, power_solution, times, title="LR and MLR", figname="Plots/Task2.png", savefig=True)


    elif Task3 == True:

        print("Task 3")

        features, target, pred_features, power_solution = Data.Data(TrainData, WF_input, Solution, meter='T3')
