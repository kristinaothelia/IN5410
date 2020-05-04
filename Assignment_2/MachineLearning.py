"""
IN5410 - Energy informatics | Assignment 2

Linear Regression           | linreg
k-Nearest Neighbor          | kNN
Supported Vector Regression | SVR
Artificial Neural Networks  | ANN
"""

import numpy  					as np
import pandas 					as pd
import readData             	as Data

from sklearn.model_selection  	import train_test_split
#from sklearn.cross_validation import train_test_split # difference?
from sklearn.preprocessing 		import StandardScaler
from sklearn.linear_model     	import LinearRegression
from sklearn.metrics       	  	import mean_squared_error, r2_score, classification_report, confusion_matrix
from sklearn.neighbors 		  	import KNeighborsClassifier
from sklearn.svm 			  	import SVR
# -----------------------------------------------------------------------------

"""
def TrainTestSplit(feature, target, TestSize=0.2):
	X_train, X_test, y_train, y_test = train_test_split(feature, target, test_size=TestSize, random_state=0)
	#return X_train, X_test, y_train, y_test
"""

def linreg(TrainData, WF_input, Solution):
	"""
	#https://towardsdatascience.com/a-beginners-guide-to-linear-regression-in-python-with-scikit-learn-83a8f7ae2b4f

	Linear regression finds the parameters to minimize the MSE between the
	predictions and the targets.
	"""

	# 'Power' is the 'target' and the other columns are the 'features'
    # Only use the features for 10m, so dropping the 100m colum
	features, target, pred_features, power_solution = Data.Data(TrainData, WF_input, Solution)

	linreg = LinearRegression(normalize=True)		# Model
	linreg.fit(features, target) 					# Training the model

	# BRUKES DETTE?
	print("intercept_ : ", linreg.intercept_)		# To retrieve the intercept
	print("coef_      : ", linreg.coef_)			# For retrieving the slope

	# Make predictions
	y_pred = linreg.predict(pred_features)

	compare_values = pd.DataFrame({'Actual': power_solution.flatten(), 'Predicted': y_pred.flatten()})
	print("Comapre power_solution and y_pred:")
	print(compare_values)

	return y_pred, power_solution


def kNN(TrainData, WF_input, Solution, k):
	#https://towardsdatascience.com/knn-using-scikit-learn-c6bed765be75

	# Tester forst helt basic

	features, target, pred_features, power_solution = Data.Data(TrainData, WF_input, Solution)

	print("\nFeatures:")
	print(features)
	print("\nTargets:")
	print(target)

	classifier = KNeighborsClassifier(n_neighbors=k)
	classifier.fit(features, target)
	y_pred = classifier.predict(pred_features)

	#print(confusion_matrix(power_solution, y_pred))
	#print(classification_report(power_solution, y_pred))
	# plot to find optimal value for k, maybe have 'find_k/k_fold' in separate function?

	return y_pred, power_solution

"""
def kNN(X_train, X_test, y_train, y_test):	# TrainData, WF_input, Solution
	#https://towardsdatascience.com/knn-using-scikit-learn-c6bed765be75

	k_list = range(1,26)
	scores = {}
	scores_list = []

	for k in k_list:
		knn = KNeighborsClassifier(n_neighbors=k)
		knn.fit(X_train, y_train)
		y_pred = knn.predict(X_test)
		scores[k] = metrics.accuracy_score(y_test, y_pred)
		scores_list.append(metrics.accuracy_score(y_test, y_pred))


	# plot to find optimal value for k, maybe have 'find_k/k_fold' in separate function?
"""

def SVR():
	#https://medium.com/pursuitnotes/support-vector-regression-in-6-steps-with-python-c4569acd062d
	svr_reg = SVR(kernel='rbf') # see link/documentation to choose kernel
	svr_reg.fit(X_train,y_train)
	y_pred = svr_reg.predict(X_test)


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
	Function to calculate the mean squared error (MSE) for our model
	Input power_solution	| Function array
	Input y_pred			| Predicted function array
	"""
	return mean_squared_error(power_solution, y_pred)

def RMSE(power_solution, y_pred):
	""" sqrt of MSE """
	return np.sqrt(mean_squared_error(power_solution, y_pred))

def ErrorTable(y_test, y_train):
	df = pd.DataFrame({'Actual': y_test.flatten(), 'Predicted': y_pred.flatten()})
