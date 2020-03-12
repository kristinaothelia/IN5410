# IN5410 - Assignment 1: Functions
# -----------------------------------------------------------------------------
import os, random, xlsxwriter, sys, random

import matplotlib.pyplot    as plt
import pandas               as pd
import numpy                as np

from datetime               import time
from scipy.optimize         import linprog
from random                 import seed
# -----------------------------------------------------------------------------

#header_names = list(df)
#print(header_names)
#print(df['Alpha']['EV'])

seed = 5410

# Funker ikke som det skal med random.uniform() !!!


def Get_df(file_name='/energy_use.xlsx'):
    """
    Function for reading the xl-file 
    """

    cwd      = os.getcwd()
    filename = cwd + file_name
    nanDict  = {}
    df       = pd.read_excel(filename, header=0, skiprows=0, index_col=0, na_values=nanDict)

    return df

def applications(df):
    """
    """

    n_app         = len(df)                     # Number of appliances
    app_names     = df.index.values

    # Get variables from the Excel file
    shiftable     = df[df['Shiftable'] == 1]    # Shiftable appliances
    non_shiftable = df[df['Shiftable'] == 0]    # Non-shiftable appliances

    alpha         = df['Alpha'].values          # Lower bounce. Set-up time
    beta          = df['Beta'].values           # Upper bounce. Deadline
    length        = df['Length'].values

    #print(str(app_names))
    return n_app, app_names, shiftable, non_shiftable, alpha, beta, length

def Get_price(hours, seed, ToU=False):
    """
    Function returning either Time of Use (ToU) or Real Time Price (RTP)
    """

    if ToU == True:
        price = [0.5] * hours
        price[17:20] = [1.0,1.0,1.0]

    else:
        # Legge inn riktig RTP price
        price        = [0.5] * hours
        price[6:9]   = [0.65,0.75, 0.65]
        price[16:20] = [0.75,1.0,1.0,1.0]

        price = np.array(price)
        
        for i in range(len(price)):
            # Outside peak-hour 
            if i < 6:
                price[i] = price[i] + random.uniform(0, 0.05)
            elif i > 17 or i < 20:
                price[i] = price[i] + random.uniform(0, 0.2)
            # During peak-hour
            else:
                price[i] = price[i] + random.uniform(0, 0.4)
        
    return list(price)


def interval(hour=1, start=0, stop=23, shuffle=False):
    """
    Creating time intervals
    """

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


def linprog_input(df, n_app, price, intervals, hours=24):
    """
    Function returning the inputs needed in linprog
    """

    c = np.array(price*len(df))

    energy_hour = df['Hourly usage [kW]'].values
    E_tot       = [np.sum(energy_hour)]*hours


    A_eq = np.zeros((n_app, n_app*hours))
    b_eq = df['Daily usage [kW]'].values

    # ????
    for i in range(n_app):
        for j in range(hours):
            A_eq[i,j+(hours*i)] = intervals[i][j]

    # ????
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

    return c, A_eq, b_eq, A_ub, b_ub


def Make_p_hist(price):
    """
    """

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

def consumption_plot(shift=None, nonshift=None, shiftnames=None, nonshiftnames=None, price=None):

    f, consumptionfig = plt.subplots(1, 1, figsize=(10,7))
    #plt.style.use("dark_background")
    
    if shift is not None:
        length = len(shift[0])
    elif nonshift is not 0:
        length = len(nonshift[0])
    elif price is not None:
        length = len(price)

    bins = np.arange(0, length)
    width = 0.9
    bottom = np.zeros(length)

    #iterate over shiftable and nonshiftable appliances to create stacked
    # bars for the chart.
    if nonshift is not 0:
        for i in range(len(nonshift)):
            consumptionfig.bar(bins, nonshift[i], width=width, bottom=bottom,label=nonshiftnames[i])
            bottom = np.add(bottom, nonshift[i])

    if shift is not None:
        for i in range(len(shift)):
            consumptionfig.bar(bins, shift[i], width=width, bottom=bottom,label=shiftnames[i])
            bottom = np.add(bottom, shift[i])

    #consumptionfig.set(title='Consumption of households',xlabel='Hour',xticks=bins,ylabel='Consumption, kWh')
    consumptionfig.set_title('Consumption of households', fontweight='bold', size=16)
    consumptionfig.set_ylabel('Consumption, kWh', fontsize=16)
    consumptionfig.set_xlabel('Hour', fontsize=16)
    consumptionfig.set(xticks=bins)

    #Making the figure pretty
    consumptionfig.tick_params(axis="both", which="both", bottom="off",
                               top="off", labelbottom="on", left="off",
                               right="off", labelleft="on")

    consumptionfig.set_axisbelow(True)
    consumptionfig.grid(b=True, which='major', axis='y', color='#cccccc',linestyle='--')

    if price is not None:
        pricefig = consumptionfig.twinx()
        pricefig.step(bins, price, color='black', where='mid', label='price')
        #pricefig.set(ylabel='Price, NOK/kWh')
        pricefig.set_ylabel('Price, NOK/kWh', fontsize=16)
        consumptionfig.set_axisbelow(True)


    #retrieving labels to make a neat legend
    handles, labels = consumptionfig.get_legend_handles_labels()
    handle, label =pricefig.get_legend_handles_labels()
    consumptionfig.legend(bbox_to_anchor=(1.125, 1), loc=2, borderaxespad=0., fontsize=15)
    pricefig.legend(bbox_to_anchor=(1.125, 0), loc=2, borderaxespad=0., fontsize=15)
    #consumptionfig.legend(bbox_to_anchor=(0.5, -0.35), loc=8, borderaxespad=0., fontsize=15)


    plt.tight_layout()
    plt.show()