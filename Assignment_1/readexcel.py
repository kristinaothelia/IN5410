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
print(df)

price = [0.5] * 24
price[17:20] = [1.0,1.0,1.0]


"""
header_names = list(df)
print(header_names)

print(df['Alpha']['EV'])

t = []

for i in range(0, 24):
	if i < 23:
		t.append(time(i, 0).isoformat(timespec='hours') + ' - ' + time(i+1, 0).isoformat(timespec='hours'))
	else:
		t.append(time(i, 0).isoformat(timespec='hours') + ' - ' + time(0, 0).isoformat(timespec='hours'))

#plt.style.use("dark_background")
plt.bar(t, price, color='g')
plt.title("Time of Use [ToU]", fontsize='16', weight='bold')
plt.ylabel("Price [NOK/kWh]", fontsize='15')
plt.xlabel("Time [UTC]", fontsize='15')
plt.gcf().autofmt_xdate(rotation=70, ha='center')
plt.show()
"""


c     = np.array(price*len(df))
alpha = df['Alpha'].values
beta  = df['Beta'].values

A_eq = np.array([[0.0]*len(c)]*len(df))   # Equality constraint
A_ub = np.array([[0.0]*len(c)]*len(c))

b_eq = df['Daily usage [kW]'].values      # Equality constraint
b_ub = np.array([[0.0]*24]*len(df))


for i in range(len(df)):
	print(df.index[i])
	for j in range(24):
		if j >= alpha[i] and j <= beta[i]:
			b_ub[i][j] = df['Hourly usage [kW]'].values[i]

			#A_ub[i][j] = 1  # ??
			#A_eq[i][j] = 1  # ??


b_ub = b_ub.ravel()  # create 1D array
print(b_ub, '--------------')

print(np.shape(A_eq))
print(np.shape(b_eq))
print(np.shape(A_ub))
print(np.shape(b_ub))
print(len(c))


res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=None)

print(res)
print(str(res.fun))
