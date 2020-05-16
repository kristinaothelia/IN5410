import os, sys

import matplotlib.pyplot 		as plt
import pandas 					as pd
import numpy  					as np
import Data             		as Data
import MachineLearning          as ML

from prettytable                import PrettyTable
from sklearn.neural_network  	import MLPRegressor
from sklearn.linear_model     	import LinearRegression
from sklearn.metrics       	  	import mean_squared_error, r2_score

# For implementing RNN
stderr = sys.stderr
sys.stderr = open(os.devnull, 'w')
#import keras
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
sys.stderr = stderr

np.random.seed(5410)


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

look_back      = 3
# Is it right to split test and solution data?
trainX, trainY = create_dataset(dataset, look_back)
testX, testY   = create_dataset(solution, look_back)

# LinReg
# -----------------------------------------------------------------------------
linreg = LinearRegression()		# Model
linreg.fit(trainX, trainY) 		# Training the model

# Make predictions
trainPredict = linreg.predict(trainX)
y_pred       = linreg.predict(testX)

# sol er fortsatt en for lang, vet ikke hvordan vi skal fikse det
metrics(testY, y_pred)

plt.figure()
plt.plot(y_pred, label="y_pred")
plt.plot(testY, label="testY")
plt.title("LR"); plt.legend()#; plt.show()

# RNN
# -----------------------------------------------------------------------------
trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))

# create and fit the LSTM network
model = Sequential()
# What should the number be? 1?
units = 3   # trainX.shape[2]
#model.add(LSTM(units, input_shape=(1, look_back)))  # activation='relu', return_sequences=True
#model.add(LSTM(units, activation='relu', input_shape=(trainX.shape[1], look_back)))

# YouTube
model.add(LSTM(4, activation='relu', return_sequences=True, input_shape=(trainX.shape[1], look_back)))
model.add(Dropout(0.2))
model.add(LSTM(20, activation='relu', return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(80, activation='relu', return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(120, activation='relu'))
model.add(Dropout(0.2))


model.add(Dense(1))
#model.summary()

model.compile(loss='mean_squared_error', optimizer='adam')
#model.fit(trainX, trainY, epochs=10, batch_size=16, verbose=2) # 32
model.fit(trainX, trainY, epochs=10, batch_size=16) # 32

# make predictions
trainPredict = model.predict(trainX)
testPredict = model.predict(testX)      # y_pred

metrics(testY, testPredict)

# plot baseline and predictions
plt.figure()
plt.plot(testY, label="testY")
plt.plot(testPredict, label="y_pred")
plt.title("RNN"); plt.legend(); plt.show()
