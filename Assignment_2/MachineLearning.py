"""
IN5410 - Energy informatics | Assignment 2

The machine learning techniques include:

Linear Regression               | linreg
k-Nearest Neighbor              | kNN
Supported Vector Regression     | SVR
Artificial Neural Networks      | ANN
"""

import numpy  as np
import pandas as pd

from sklearn import metrics
from sklearn.model_selection  import train_test_split
#from sklearn.cross_validation import train_test_split # difference?
from sklearn.linear_model     import LinearRegression
from sklearn.neighbors 		  import KNeighborsClassifier
from sklearn.svm 			  import SVR
# -----------------------------------------------------------------------------

def TrainTestSplit(feature, target, TestSize=0.2):
	X_train, X_test, y_train, y_test = train_test_split(feature, target, test_size=TestSize, random_state=0)
	#return X_train, X_test, y_train, y_test

def linreg(X_train, X_test, y_train):
	#https://towardsdatascience.com/a-beginners-guide-to-linear-regression-in-python-with-scikit-learn-83a8f7ae2b4f
	linreg = LinearRegression()
	linreg.fit(X_train, y_train) #training the algorithm

	#To retrieve the intercept:
	print(linreg.intercept_)
	#For retrieving the slope:
	print(linreg.coef_)

	y_pred = linreg.predict(X_test)


def kNN(X_train, X_test, y_train, y_test):
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

def SVR():
	#https://medium.com/pursuitnotes/support-vector-regression-in-6-steps-with-python-c4569acd062d
	svr_reg = SVR(kernel='rbf') # see link/documentation to choose kernel
	svr_reg.fit(X_train,y_train)
	y_pred = svr_reg.predict(X_test)

def ANN():
    pass

def ErrorTable(y_test, y_train):
	df = pd.DataFrame({'Actual': y_test.flatten(), 'Predicted': y_pred.flatten()})
