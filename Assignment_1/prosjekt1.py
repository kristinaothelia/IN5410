import os, random, xlsxwriter, sys


import matplotlib.pyplot   as plt
import numpy               as np
import pandas              as pd


from	datetime	   import  time
from    scipy.optimize import  linprog

# Python 3.7.4
#------------------------------------------------------------------------------
cwd      = os.getcwd()
filename = cwd + '/energy_use.xlsx'
nanDict  = {}
df       = pd.read_excel(filename, header=0, skiprows=0, index_col=0, na_values=nanDict)

df = df[-3:]  # only look at the 3 last appliances
#print(df)

hours = 24
n_app = len(df)  # number of appliances

price = [0.5] * hours
price[17:20] = [1.0,1.0,1.0]

shiftable     = df[df['Shiftable'] == 1]
non_shiftable = df[df['Shiftable'] == 0]

alpha = df['Alpha'].values
beta  = df['Beta'].values


def interval(hour=1, start=0, stop=23, shuffle=False):

	if shuffle:
		interval = np.zeros(24)
		interval[:hour] = 1
		np.random.shuffle(interval)

	else:
		interval = np.zeros(stop-start + 1)
		interval[:hour] = 1
		#np.random.shuffle(interval)
		if start == 0:
			padLeft = np.zeros(0)
		else:
			padLeft = np.zeros(start)
		if stop == 23:
			return np.append(padLeft,interval)
		else:
			padRight = np.zeros(24-len(interval)-len(padLeft))
			return np.append(np.append(padLeft,interval),padRight)

	return interval


intervals = []
for i in range(n_app):
	intervals.append(interval(hour=3, start=alpha[i], stop=beta[i], shuffle=False))

intervals = np.array(intervals)
print(intervals)

c = np.array(price*len(df))

energy_hour = df['Hourly usage [kW]'].values
E_tot       = [np.sum(energy_hour)]*hours


A_eq = np.zeros((n_app, n_app*hours))  
b_eq = df['Daily usage [kW]'].values

for i in range(n_app):
	for j in range(hours):
		A_eq[i,j+(hours*i)] = intervals[i][j]


A_mul = np.zeros((hours,n_app*hours))
for i in range(A_mul.shape[0]):
	A_mul[i,i::hours] = 1
A_one = np.eye(n_app*hours)
A_ub  = np.concatenate((A_one,A_mul),axis=0)

b = []
for i in energy_hour:
	b.append([i]*hours)
b = np.array(b).ravel()
b_ub = np.concatenate((b, E_tot))


#print(np.shape(A_eq))
#print(np.shape(b_eq))
#print(np.shape(A_ub))
#print(np.shape(b_ub))
#print(len(c))


res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=(0,None))

print(res)
print(str(res.fun))

