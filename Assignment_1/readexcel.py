import os, random, xlsxwriter


import matplotlib.pyplot   as plt
import numpy               as np
import pandas              as pd


from	datetime	import	 time

# Python 3.7.4
#------------------------------------------------------------------------------
cwd      = os.getcwd()
filename = cwd + '/energy_use.xlsx'
nanDict  = {}
df       = pd.read_excel(filename, header=0, skiprows=0, index_col=0, na_values=nanDict)

print(df)


header_names = list(df)
print(header_names)

print(df['Alpha']['EV'])

t = []

for i in range(0, 24):
	if i < 23:
		t.append(time(i, 0).isoformat(timespec='hours') + ' - ' + time(i+1, 0).isoformat(timespec='hours'))
	else:
		t.append(time(i, 0).isoformat(timespec='hours') + ' - ' + time(0, 0).isoformat(timespec='hours'))

print(t)

price = [0.5] * 24
price[17:20] = [1.0,1.0,1.0]

print(price)

#plt.style.use("dark_background")
plt.bar(t, price, color='g')
plt.title("Time of Use [ToU]", fontsize='16', weight='bold')
plt.ylabel("Price [NOK/kWh]", fontsize='15')
plt.xlabel("Time [UTC]", fontsize='15')
plt.gcf().autofmt_xdate(rotation=70, ha='center')
plt.show()



