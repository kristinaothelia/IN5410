"""
IN5410 - Energy informatics | Assignment 2

Linear Regression           | linreg
k-Nearest Neighbor          | kNN
Supported Vector Regression | SVR
Feedforward Neural Networks | FFNN
Recurrent Neural Networks  	| RNN

We use the following train/test data for Task 1 and 2 Machine Learning methods:
X_train						| features
X_test						| pred_features
y_train						| target
y_test						| power_solution
"""

import os, sys

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

print("test")
# For implementing RNN
stderr = sys.stderr
sys.stderr = open(os.devnull, 'w')
#import keras
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout, SimpleRNN
sys.stderr = stderr


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
	#compare_values = pd.DataFrame({'Actual': power_solution.flatten(), 'Predicted': y_pred.flatten()})
	#print("\nComapre power_solution and y_pred:\n", compare_values)

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

def FFNN_gridsearch(features, target, pred_features, power_solution, lmbd_vals, eta_vals, shuffle=False):
	""" Finding the best parameters using GridSearchCV """

	parameters  = {	'activation': ['relu', 'logistic'], \
					'solver': ['sgd', 'adam'],			\
					'alpha': lmbd_vals,					\
					'learning_rate': ['constant', 'adaptive'], \
					'learning_rate_init': eta_vals,		\
					'max_iter': [500, 1000, 1500]}

	ffnn 		= MLPRegressor(shuffle=shuffle)
	grid_search = GridSearchCV(ffnn, parameters, n_jobs=-1)
	grid_search.fit(features, target.ravel())

	best_params = grid_search.best_params_
	print("\nBest parameters: ", best_params)

	return best_params

def FFNN_Heatmap_MSE_R2(features, target, pred_features, power_solution, eta_vals, lmbd_vals, shuffle=False):
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
			    				tol=1e-5,
								shuffle=shuffle )

			reg.fit(features, target)
			y_pred    = reg.predict(pred_features)

			MSE_[i][j]  = MSE(power_solution, y_pred)
			RMSE_[i][j] = RMSE(power_solution, y_pred)
			R2_[i][j]   = r2_score(power_solution, y_pred)

			# This can probably be taken away later, or insert in a 'if print=True'?

			print("Learning rate = ", eta)
			print("Lambda =        ", lmbd)
			print("MSE score:      ", MSE(power_solution, y_pred))
			print("RMSE score:     ", RMSE(power_solution, y_pred))
			print("R2 score:       ", r2_score(power_solution, y_pred))
			print()


	#etas = ["{:0.2e}".format(i) for i in eta_vals]
	return MSE_, RMSE_, R2_

def FFNN(features, target, pred_features, power_solution, lmbd_vals, eta_vals, default=False, shuffle=True):
	"""
	Feed Forward Neural Network
	# Best parameters:  {'activation': 'relu', 'alpha': 0.001, 'learning_rate': 'adaptive', 'learning_rate_init': 0.1, 'max_iter': 1500, 'solver': 'sgd'}
	# Best parameters:  {'activation': 'relu', 'alpha': 0.0001, 'learning_rate': 'adaptive', 'learning_rate_init': 0.1, 'max_iter': 500, 'solver': 'sgd'}
	"""
	if default:

		activation    		= "relu"
		solver		  		= "adam" 		# sdg
		learning_rate 		= "adaptive"	# constant
		alpha 		  		= 0.001
		learning_rate_init  = 1e-5

		reg = MLPRegressor(	activation=activation, solver=solver,
							learning_rate=learning_rate,
							alpha=alpha, learning_rate_init=learning_rate_init,
							max_iter=1000, tol=1e-5, shuffle=shuffle )
	else:
		best_params   		= FFNN_gridsearch(features, target, pred_features, power_solution, lmbd_vals, eta_vals, shuffle=shuffle)
		reg 		  		= MLPRegressor().set_params(**best_params)
		activation    		= best_params['activation']
		solver		  		= best_params['solver']
		alpha         		= best_params['alpha']
		learning_rate_init 	= best_params['learning_rate_init']


	reg    = reg.fit(features, target)        # Training the model
	y_pred = reg.predict(pred_features)       # Predicting

	# Compare predicted and actual values
	#compare_values = pd.DataFrame({'Actual': power_solution.flatten(), 'Predicted': y_pred.flatten()})
	#print("\nComapre power_solution and y_pred:\n", compare_values)

	return y_pred, power_solution, activation, solver, alpha, learning_rate_init


def RNN_gridsearch(features, target, pred_features, power_solution):
	""" Finding the best parameters using GridSearchCV """
	pass

def RNN(look_back, trainX, trainY, testX, testy):
	# https://www.artificiallyintelligentclaire.com/recurrent-neural-networks-python/
	# https://machinelearningmastery.com/time-series-prediction-lstm-recurrent-neural-networks-python-keras/
	# Installing keras: https://anaconda.org/conda-forge/keras

	units=4
	epo = 10
	bz=10
	# create and fit the LSTM network
	model = Sequential()
	model.add(LSTM(units=units, input_shape=(1, look_back)))  # return_sequences=True #stateful=True
	model.add(Dense(1))                           
	model.compile(loss='mean_squared_error', optimizer='adam')
	#model.fit(trainX, trainY, epochs=epo, batch_size=1, verbose=2)

	history = model.fit(trainX, trainY, epochs=epo, batch_size=bz, validation_data=(testX, testy), verbose=2, shuffle=False)

	# plot history
	plt.plot(history.history['loss'], label='train')
	plt.plot(history.history['val_loss'], label='test')
	plt.legend()
	plt.show()

	print(model.summary())

	# make predictions
	trainPredict = model.predict(trainX)
	testPredict  = model.predict(testX)

	'''
	# Annet eksempel:
	# Initilizing the RNN
	reg = Sequential()

	# Adding the first LSTM layer and some Dropout regularization
	reg.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1],1)))
	reg.add(Dropout(0.2))

	# Adding a second LSTM layer and some Dropout regularization
	reg.add(LSTM(units=50, return_sequences = True))
	reg.add(Dropout(0.2))

	# Adding a third LSTM layer and some Dropout regularization
	reg.add(LSTM(units=50, return_sequences = True))
	reg.add(Dropout(0.2))

	# Adding a fourth LSTM layer and some Dropout regularization
	reg.add(LSTM(units=50))
	reg.add(Dropout(0.2))

	# Adding the output layer
	reg.add(Dense(units=1))
	'''
	return trainPredict, testPredict



def LR_SVR(trainX, trainY, testX, testY):

	"""
	def timeseries_to_supervised(data, lag=1):
		'''
		Transformes the timeseries data into supervised dataset
								 [a1, a2]
								 [a2, a3]
		[a1,a2,a3,a4,a5,a6] ---> [a3, a4]
								 [a4, a5]
								 [a5, a6]
								 [a6,  0]
		'''
		df = pd.DataFrame(data)
		columns = [df.shift(i) for i in range(1, lag+1)]
		columns.append(df)
		df = pd.concat(columns, axis=1)
		df.fillna(0, inplace=True)
		return df
	"""

	""" Noe aa bruke?
	for i in range(len(solution_power)):
		#print(np.array([previous_power]).shape)
		next_hour_prediction = model.predict(np.array([previous_power]))[0][0]
		#print(next_hour_prediction)
		power_prediction.append(next_hour_prediction)

		previous_power = previous_power[1:]
		previous_power.append(next_hour_prediction)
	"""

	#supervised          = timeseries_to_supervised(target, 1)
	#supervised_values   = supervised.values

	#trainX = supervised_values[:,0].reshape(1, -1)
	#trainY = supervised_values[:,1].reshape(1, -1)
	#testX  = power_solution

	y_pred_LR, power_solution = linreg(features=trainX, target=trainY, pred_features=testX, power_solution=testY)	# ??
	y_pred_SVR, power_solution, kernel, C, gamma, epsilon = SVR_func(features=trainX, target=trainY, pred_features=testX, power_solution=testY)
	#trainPredict, testPredict = linear_regression_T3(X_train=trainX, y_train=trainY, X_test=testX)
	return y_pred_LR, y_pred_SVR, power_solution




def FFNN_RNN():
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
