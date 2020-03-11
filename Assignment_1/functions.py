# IN5410 - Assignment 1: Functions
# -----------------------------------------------------------------------------
import os, random, xlsxwriter, sys

import matplotlib.pyplot    as plt
import pandas               as pd
import numpy                as np

from datetime               import  time
from scipy.optimize         import  linprog
# -----------------------------------------------------------------------------

def Get_df(file_name='/energy_use.xlsx'):

    cwd      = os.getcwd()
    filename = cwd + file_name
    nanDict  = {}
    df       = pd.read_excel(filename, header=0, skiprows=0, index_col=0, na_values=nanDict)

    return df

def Get_price(hours, ToU=False):

    if ToU == True:
        price = [0.5] * hours
        price[17:20] = [1.0,1.0,1.0]
    else:
        # Legge inn riktig RPT price
        price = [0.5] * hours
        price[17:20] = [1.0,1.0,1.0]

    return price


#header_names = list(df)
#print(header_names)

#print(df['Alpha']['EV'])

def Make_p_hist(price):

    t = []
    for i in range(0, 24):
    	if i < 23:
    		t.append(time(i, 0).isoformat(timespec='hours') + ' - ' + time(i+1, 0).isoformat(timespec='hours'))
    	else:
    		t.append(time(i, 0).isoformat(timespec='hours') + ' - ' + time(0, 0).isoformat(timespec='hours'))
    print(t)
    print(price)
    #plt.style.use("dark_background")
    plt.bar(t, price, color='g')
    plt.title("Time of Use [ToU]", fontsize='16', weight='bold')
    plt.ylabel("Price [NOK/kWh]", fontsize='15')
    plt.xlabel("Time [UTC]", fontsize='15')
    plt.gcf().autofmt_xdate(rotation=70, ha='center')
    plt.show()
