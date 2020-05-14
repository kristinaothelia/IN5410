import os, sys

import matplotlib.pyplot 		as plt
import pandas 					as pd
import numpy  					as np
import Data             		as Data
import MachineLearning          as ML

from prettytable                import PrettyTable
from sklearn.model_selection 	import GridSearchCV
from sklearn.neural_network  	import MLPRegressor
from sklearn.preprocessing 		import StandardScaler
from sklearn.linear_model     	import LinearRegression
from sklearn.metrics       	  	import mean_squared_error, r2_score
from sklearn.neighbors 		  	import KNeighborsRegressor
from sklearn.svm 			  	import SVR

# For implementing RNN
stderr = sys.stderr
sys.stderr = open(os.devnull, 'w')
#import keras
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
sys.stderr = stderr

np.random.seed(5410)

###
# Men dette er LSTM network.. Har det noe aa si?
###


# convert an array of values into a dataset matrix
def create_dataset(dataset, look_back=1):
    dataX, dataY = [], []
    #for i in range(len(dataset)-look_back-1):  # original fra eks
    for i in range(len(dataset)-look_back):
        a = dataset[i:(i+look_back), 0]
        dataX.append(a)
        dataY.append(dataset[i + look_back, 0])
    return np.array(dataX), np.array(dataY)

def metrics(power_solution, y_pred):
    x = PrettyTable()
    x.field_names = ["MSE", "RMSE", "R2"]
    x.add_row(["%.3f"% ML.MSE(power_solution, y_pred), "%.3f"% ML.RMSE(power_solution, y_pred), "%.3f"% ML.R2(power_solution, y_pred)])
    print(x)

# -----------------------------------------------------------------------------

dataset  = Data.Get_data(filename='/Data/TrainData.csv')
solution = Data.Get_data(filename='/Data/Solution.csv')
dataset  = dataset.loc[:, dataset.columns == 'POWER'].values    # 16080
solution = solution.loc[:, solution.columns == 'POWER'].values  # 720

# Dette burde vi ikke trenge
#scaler = StandardScaler()
#dataset = scaler.fit_transform(dataset)

"""
# split into train and test sets
# NB!!! Skal vi gjore dette..????????????

train_size = int(len(dataset) * 0.67)
test_size = len(dataset) - train_size
train, test = dataset[0:train_size,:], dataset[train_size:len(dataset),:]
"""
look_back = 3
trainX, trainY = create_dataset(dataset, look_back)
testX, testY = create_dataset(solution, look_back)


# LinReg
# -----------------------------------------------------------------------------
linreg = LinearRegression()		# Model
linreg.fit(trainX, trainY) 					# Training the model

# Make predictions
trainPredict = linreg.predict(trainX)
y_pred = linreg.predict(testX)

# sol er fortsatt en for lang, vet ikke hvordan vi skal fikse det
metrics(testY, y_pred)

plt.figure()
plt.plot(y_pred, label="y_pred")
plt.plot(testY, label="testY")
plt.title("LR")
plt.legend()#; plt.show()

# RNN
# -----------------------------------------------------------------------------

trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))

# create and fit the LSTM network
model = Sequential()
model.add(LSTM(4, input_shape=(1, look_back)))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(trainX, trainY, epochs=3, batch_size=1, verbose=2)


# make predictions
trainPredict = model.predict(trainX)
testPredict = model.predict(testX)
# invert predictions
#trainPredict = scaler.inverse_transform(trainPredict)
#trainY = scaler.inverse_transform([trainY])
#testPredict = scaler.inverse_transform(testPredict)
#testY = scaler.inverse_transform([testY])

metrics(testY, testPredict)

""" Disse linjene virker ikke...
# shift test predictions for plotting
testPredictPlot = np.empty_like(dataset)
testPredictPlot[:, :] = np.nan
testPredictPlot[len(trainPredict)+(look_back*2)+1:len(dataset)-1, :] = testPredict
"""
# plot baseline and predictions
plt.figure()
plt.plot(testY, label="testY")
plt.plot(testPredict, label="y_pred")
plt.title("RNN")
plt.legend(); plt.show()
