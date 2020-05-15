"""
IN5410 - Energy informatics | Assignment 2
"""
import MachineLearning      as ML
# -----------------------------------------------------------------------------

def Task1_kNN_input(features, target, pred_features, power_solution):

    input_ = int(input("Do you want to: \n1) Use predefined values based on default- and GridSearchCV values? (Time efficient option) \n2) Use the 'best parameters' from GridSearchCV? \n3) Enter your own parameters based on a n_neighbors (k) plot? \nEnter 1, 2 or 3 (int): "))

    if input_ == 1:
        y_pred, power_solution, k, weights, p = ML.kNN(features, target, pred_features, power_solution, default=True)
    elif input_ == 2:
        best_params = ML.kNN_gridsearch(features, target, pred_features, power_solution)
        y_pred, power_solution, k, weights, p = ML.kNN(features, target, pred_features, power_solution, best_params, BestParams=True)
    elif input_ == 3:
        best_params = ML.kNN_gridsearch(features, target, pred_features, power_solution, plot=True)
        y_pred, power_solution, k, weights, p = ML.kNN(features, target, pred_features, power_solution, best_params, BestParams=False)
    else:
        print("Enter 1, 2 or 3, then enter"); exit()

    return y_pred, power_solution, k, weights, p


def Task1_SVR_input(features, target, pred_features, power_solution):

    input_ = int(input("Do you want to: \n1) Use predefined values based on default- and GridSearchCV values? (Time efficient option) \n2) Use the 'best parameters' from GridSearchCV? \nEnter 1 or 2 (int): "))

    if input_ == 1:
        y_pred, power_solution, kernel, C, gamma, epsilon = ML.SVR_func(features, target, pred_features, power_solution)
    elif input_ == 2:
        y_pred, power_solution, kernel, C, gamma, epsilon = ML.SVR_func(features, target, pred_features, power_solution, default=False)
    else:
        print("Enter 1 or 2, then enter"); exit()

    return y_pred, power_solution, kernel, C, gamma, epsilon


def Task1_ANN_input(features, target, pred_features, power_solution):
    pass
