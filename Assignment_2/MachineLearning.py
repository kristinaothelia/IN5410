"""
IN5410 - Energy informatics | Assignment 2

Linear Regression           | linreg
k-Nearest Neighbor          | kNN
Supported Vector Regression | SVR
Artificial Neural Networks  | ANN
"""
import sys
import numpy  					as np
import pandas 					as pd
import readData             	as Data
import matplotlib.pyplot		as plt

from sklearn.model_selection 	import GridSearchCV
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

	# BRUKES DETTE? Var bare fra det eksempelet i adressen over, vet ikke om vi trenger det ######
	print("intercept_ : ", linreg.intercept_)		# To retrieve the intercept
	print("coef_      : ", linreg.coef_)			# For retrieving the slope

	# Make predictions
	y_pred = linreg.predict(pred_features)

	compare_values = pd.DataFrame({'Actual': power_solution.flatten(), 'Predicted': y_pred.flatten()})
	print("Comapre power_solution and y_pred:")
	print(compare_values)

	return y_pred, power_solution


def kNN_parameters(features, target, pred_features, power_solution):
	""" Finding the best parameters using GridSearchCV """

	k_range    = np.arange(1, 16, 1)
	parameters = {'n_neighbors': k_range, 'weights': ['uniform', 'distance']}

	knn  = KNeighborsRegressor()
	gscv = GridSearchCV(knn, parameters)
	gscv.fit(features, target)

	print("Best parameters: ",gscv.best_params_)

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

def kNN(features, target, pred_features, power_solution, k, weights):
	#regression example:
	#https://medium.com/analytics-vidhya/k-neighbors-regression-analysis-in-python-61532d56d8e4

	knn 	= KNeighborsRegressor(n_neighbors=k, weights=weights)
	knn.fit(features, target)
	y_pred 	= knn.predict(pred_features)

	return y_pred, power_solution

def SVR_parameters(features, target, pred_features, power_solution):
	""" Finding the best parameters using GridSearchCV """
	pass

def SVR_func(TrainData, WF_input, Solution):
	#https://medium.com/pursuitnotes/support-vector-regression-in-6-steps-with-python-c4569acd062d
	features, target, pred_features, power_solution = Data.Data(TrainData, WF_input, Solution)
	svr_reg = SVR() # kernel='rbf' see link/documentation to choose kernel
	svr_reg.fit(features, target)
	y_pred = svr_reg.predict(pred_features)
	return y_pred, power_solution

def ANN_parameters(features, target, pred_features, power_solution):
	""" Finding the best parameters using GridSearchCV """
	pass

def ANN():
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
	""" sqrt of MSE. Value closer to 0 are better """
	return np.sqrt(mean_squared_error(power_solution, y_pred))

def ErrorTable(y_test, y_train):
	df = pd.DataFrame({'Actual': y_test.flatten(), 'Predicted': y_pred.flatten()})
