"""
IN5410 - Energy informatics | Assignment 2

Linear Regression           | linreg
k-Nearest Neighbor          | kNN
Supported Vector Regression | SVR
Artificial Neural Networks  | ANN

We use the following train/test data for all Machine Learning methods:
X_train						| features
X_test						| pred_features
y_train						| target
y_test						| power_solution
"""

import sys

import matplotlib.pyplot 		as plt
import pandas 					as pd
import numpy  					as np
import Data             		as Data

from sklearn.model_selection 	import GridSearchCV
from sklearn.neural_network  	import MLPRegressor
from sklearn.preprocessing 		import StandardScaler
from sklearn.linear_model     	import LinearRegression
from sklearn.metrics       	  	import mean_squared_error, r2_score
from sklearn.neighbors 		  	import KNeighborsRegressor
from sklearn.svm 			  	import SVR

# -----------------------------------------------------------------------------


def linreg(features, target, pred_features, power_solution):
	"""
	#https://towardsdatascience.com/a-beginners-guide-to-linear-regression-in-python-with-scikit-learn-83a8f7ae2b4f

	Linear regression finds the parameters to minimize the MSE between the
	predictions and the targets.
	"""

	linreg = LinearRegression(normalize=True)		# Model
	linreg.fit(features, target) 					# Training the model

	#print("intercept_ : ", linreg.intercept_)		# To retrieve the intercept
	#print("coef_      : ", linreg.coef_)			# For retrieving the slope

	# Make predictions
	y_pred = linreg.predict(pred_features)

	# Compare predicted and actual values
	compare_values = pd.DataFrame({'Actual': power_solution.flatten(), 'Predicted': y_pred.flatten()})
	print("\nComapre power_solution and y_pred:\n", compare_values)

	return y_pred, power_solution

def kNN_gridsearch(features, target, pred_features, power_solution, plot=False):
	""" Finding the best parameters using GridSearchCV """

	k_range	   = [2, 5, 10, 50, 100, 200, 300, 400, 500, 600, 850, 1000]
	parameters = {'n_neighbors': k_range, 'weights': ['uniform', 'distance'], 'p': [1,2]}

	knn  = KNeighborsRegressor()
	gscv = GridSearchCV(knn, parameters)
	gscv.fit(features, target)

	best_params = gscv.best_params_
	print("\nBest parameters: ", best_params)

	if plot:

		# Make error plot, to find best k-value
		r2   = []
		rmse = []

		for i in k_range:
			knn = KNeighborsRegressor(n_neighbors=i)
			knn.fit(features, target)
			y_pred = knn.predict(pred_features)
			r2.append(R2(power_solution, y_pred))
			rmse.append(RMSE(power_solution, y_pred))

		plt.plot(k_range, r2, "b",  marker='.', label="r2")
		plt.plot(k_range, rmse, "g", marker='*', label="rmse")
		plt.title('k-value dependent on r2 and RMSE \nWant high r2 score and low RMSE score')
		plt.xlabel('K Value')
		plt.ylabel('r2, RMSE')
		plt.tight_layout(), plt.legend(), plt.grid(), plt.show()

	return best_params

def kNN_parameters():
	k       = int(input("Enter wanted k value (int): "))
	p       = int(input("Enter wanted p value (int): "))
	weights = int(input("Enter wanted weights method (1 for 'uniform' or 2 for 'distance'): "))

	if weights == 1:
		weights = 'uniform'
	elif weights == 2:
		weights = 'distance'
	else:
		print("Wrong weights input, enter 1 or 2 (int)"); exit()
	return k, p, weights

def kNN(features, target, pred_features, power_solution, best_params=None, BestParams=False, default=False):
	#regression example:
	#https://medium.com/analytics-vidhya/k-neighbors-regression-analysis-in-python-61532d56d8e4

	if default:
		k, weights, p = 150, 'uniform', 1
		knn = KNeighborsRegressor(n_neighbors=k, weights=weights, p=p)

	elif BestParams:
		knn 	= KNeighborsRegressor().set_params(**best_params)
		k 		= best_params['n_neighbors']
		p 		= best_params['p']
		weights = best_params['weights']

	else:
		k, p, weights = kNN_parameters()
		knn = KNeighborsRegressor(n_neighbors=k, weights=weights, p=p)

	knn.fit(features, target)
	y_pred 	= knn.predict(pred_features)

	return y_pred, power_solution, k, weights, p

def SVR_gridsearch(features, target, pred_features, power_solution):
	""" Finding the best parameters using GridSearchCV """

	parameters  = {	'kernel': ['rbf', 'linear', 'sigmoid'], 	\
					'C': [0.001, 0.01, 0.1, 1.0], 			\
					'gamma' : ['scale', 'auto'], 	\
					'epsilon':[0.01, 0.1, 1.0]}

	svr 		= SVR()
	grid_search = GridSearchCV(svr, parameters, n_jobs=-1)
	grid_search.fit(features, target.ravel())

	best_params = grid_search.best_params_
	print("\nBest parameters: ", best_params)

	return best_params

def SVR_func(features, target, pred_features, power_solution, default=True):
	#https://medium.com/pursuitnotes/support-vector-regression-in-6-steps-with-python-c4569acd062d

	if default:
		kernel	= 'rbf'
		C		= 0.01
		gamma	= 'scale'
		epsilon = 0.1
		svr_reg = SVR(kernel=kernel, C=C, gamma=gamma, epsilon=epsilon)

	else:
		best_params = SVR_gridsearch(features, target, pred_features, power_solution)
		svr_reg = SVR().set_params(**best_params) # kernel='rbf' see link/documentation to choose kernel
		kernel	= best_params['kernel']
		C		= best_params['C']
		gamma	= best_params['gamma']
		epsilon = best_params['epsilon']

	svr_reg.fit(features, target)
	y_pred = svr_reg.predict(pred_features)

	return y_pred, power_solution, kernel, C, gamma, epsilon


def FFNN_gridsearch(features, target, pred_features, power_solution):
	""" Finding the best parameters using GridSearchCV """
	pass


def FFNN_Heatmap_MSE_R2(features, target, pred_features, power_solution, eta_vals, lmbd_vals):
	"""
	Creating MSE, RMSE and R2 values for a heatmap illustrating the best value 
	Just used what we did in fys-stk first, maybe we can use gridsearch instead
	"""

	epochs      = 500 #1000

	MSE_         = np.zeros((len(eta_vals), len(lmbd_vals)))
	RMSE_        = np.zeros((len(eta_vals), len(lmbd_vals)))
	R2_          = np.zeros((len(eta_vals), len(lmbd_vals)))


	for i, eta in enumerate(eta_vals):
		for j, lmbd in enumerate(lmbd_vals):

			reg = MLPRegressor(	activation="relu", # Eller en annen?
			    				solver="sgd",      # Eller en annen?
								alpha=lmbd,
			    				learning_rate_init=eta,
			    				max_iter=epochs,
			    				tol=1e-5 )

			reg.fit(features, target)
			y_pred    = reg.predict(pred_features)  

			MSE_[i][j]  = MSE(power_solution, y_pred)
			RMSE_[i][j] = RMSE(power_solution, y_pred)
			R2_[i][j]   = r2_score(power_solution, y_pred)

			# This can probably be taken away later, or insert in a 'if print=True'?
			print("Learning rate = ", eta)
			print("Lambda =        ", lmbd)
			print("MSE score:      ",  MSE(power_solution, y_pred))
			print("RMSE score:      ", RMSE(power_solution, y_pred))
			print("R2 score:       ",  r2_score(power_solution, y_pred))
			print()

	#etas = ["{:0.2e}".format(i) for i in eta_vals]

	return MSE_, RMSE_, R2_

def FFNN(features, target, pred_features, power_solution):
	"""
	Feed Forward Neural Network
	"""
	
	# Need to update these to the best values
	lamb  = 1e-4
	eta   = 1e-2

	reg = MLPRegressor(	activation="relu",         # Eller en annen?
						solver="sgd",              # Eller en annen?
						learning_rate='constant',
						alpha=lamb,
						learning_rate_init=eta,
						max_iter=1000,
						tol=1e-5 )

	reg    = reg.fit(features, target)        # Training the model
	y_pred = reg.predict(pred_features)       # Predicting

	# Compare predicted and actual values
	compare_values = pd.DataFrame({'Actual': power_solution.flatten(), 'Predicted': y_pred.flatten()})
	print("\nComapre power_solution and y_pred:\n", compare_values)

	return y_pred, power_solution


def RNN_gridsearch(features, target, pred_features, power_solution):
	""" Finding the best parameters using GridSearchCV """
	pass

def RNN(features, target, pred_features, power_solution):
    pass

# -----------------------------------------------------------------------------

def R2(power_solution, y_pred):
	"""
	Function to calculate the R2 score for our model. Values range from 0 to 1.
	Higher values indicate a model that is highly predictive.
	Input power_solution	| Function array
	Input y_pred			| Predicted function array
	"""
	return r2_score(power_solution, y_pred)

def MSE(power_solution, y_pred):
	"""
	Function to calculate the mean squared error (MSE) for our model.
	Value closer to 0 are better
	Input power_solution	| Function array
	Input y_pred			| Predicted function array
	"""
	return mean_squared_error(power_solution, y_pred)

def RMSE(power_solution, y_pred):
	""" sqrt of MSE. Value closer to 1 are better """
	return np.sqrt(mean_squared_error(power_solution, y_pred))

def ErrorTable(y_test, y_train):
	df = pd.DataFrame({'Actual': y_test.flatten(), 'Predicted': y_pred.flatten()})
