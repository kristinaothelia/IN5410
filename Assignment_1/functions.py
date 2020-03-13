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
    Function for reading the xlxs-file
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
        price        = [0.5] * hours            # General energy price
        price[17:20] = [1.0, 1.0, 1.0]          # After-work peak

    else:
        # RTP price
        price        = [0.5] * hours            # General energy price
        price[6:9]   = [0.65, 0.75, 0.65]       # Morning peak
        price[16:20] = [0.65, 1.0, 1.0, 1.0]    # After-work peak. Extended compared to Task 1

        price = np.array(price)

        for i in range(len(price)):
            # Outside peak-hour. Small variations during the night
            if i < 6:
                price[i] = price[i] + random.uniform(0, 0.05)
            elif i < 17 or i > 20:
                price[i] = price[i] + random.uniform(0, 0.2)
            # During peak-hour, some higher random variation
            else:
                price[i] = price[i] + random.uniform(0, 0.4)

    return list(price)

def interval(hour=1, alpha=0, beta=23, shuffle=False):
    """
    Creating a time interval

    alpha       | Setup Time
    beta        | Deadline Time
    hour        | Amount of time the app have to be active

    Her vi burde legge inn at 1 legges der det er lavest pris
    Og at den ikke kan starte for sent..
    """

    # Make a time-interval (list) from alpha and beta constrictions
    interval = np.zeros(beta-alpha + 1)

    # Sets n-hours (hour) to 1 inside the interval list. Can be set random within the interval list
    interval[:hour] = 1
    #np.random.shuffle(interval)

    if alpha == 0:
        # For the setup time, we have to move 'forward' in the time scheme --> L
        # Make an empty list, we are stepping 0 hours to the left
        Move_L = np.zeros(0)
    else:
        # Make an list of zeros in len(alpha)
        # We are moving alpha-steps/hours to the left
        Move_L = np.zeros(alpha)

    # Finish the full 24h-interval. Return the complete interval to a list
    if beta == 23:
        # If the deadline is 23, we have filled the 24h-time-slots.
        # We only need to merge the Move_L list and the interval list,
        # which are appended to a list in the main program
        return np.append(Move_L, interval)
    else:
        # If the deadline is less that 23, we have to fill the rest of the
        # 24h-time-slots with zeros, since the app cannot run here.
        # We have to fill 24 - (interval + Move_L) slots at the end
        Move_R = np.zeros(24 - (len(interval) + len(Move_L)))
        return np.append(np.append(Move_L, interval), Move_R)

    return interval

def interval_severeal_app(n_app, length, alpha, beta, shuffle=False):
    """
    Make intervals for several appliances
    """
    intervals = []
    for i in range(n_app):
        intervals.append(interval(hour=length[i], alpha=alpha[i], beta=beta[i], shuffle=False))

    return np.array(intervals)

def linprog_input(df, n_app, price, intervals, hours=24):
    """
    Function returning the inputs needed in linprog
    c       | Write what they are
    A_eq    |
    b_eq    | Daily usage [kW]
    A_ub    |
    b_ub    |
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


def Make_p_hist(df, price):
    """
    Generate a histogram of the pricing scheme
    """

    t = []
    for i in range(0, 24):
        if i < 23:
            t.append(time(i, 0).isoformat(timespec='hours') + ' - ' + time(i+1, 0).isoformat(timespec='hours'))
        else:
            t.append(time(i, 0).isoformat(timespec='hours') + ' - ' + time(0, 0).isoformat(timespec='hours'))

    plt.bar(t, price, color='g')
    plt.ylabel("Price [NOK/kWh]", fontsize='15')
    plt.xlabel("Time [UTC]", fontsize='15')
    plt.gcf().autofmt_xdate(rotation=70, ha='center')

    if len(df) == 3:
        plt.title("Time of Use [ToU]", fontsize='16', weight='bold')
        plt.savefig("Task1_hist.png")
    else:
        plt.title("Time of Use [RTP]", fontsize='16', weight='bold')
        plt.savefig("Task2_hist.png")

    plt.show()


def consumption_plot(shift=None, nonshift=None, shiftnames=None, nonshiftnames=None, price=None):
    """
    Generate a histogram plotof the appliance consumption, including
    a graphical line of the pricing scheme
    """
    f, consumptionfig = plt.subplots(1, 1, figsize=(10,6))

    if shift is not None:
        length = len(shift[0])
    elif nonshift is not 0:
        length = len(nonshift[0])
    elif price is not None:
        length = len(price)

    bins    = np.arange(0, length)
    width   = 0.9
    bottom  = np.zeros(length)

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
    consumptionfig.set_title('Household consumption ', fontweight='bold', size=16)
    consumptionfig.set_ylabel('Consumption [kWh]', fontsize=16)
    consumptionfig.set_xlabel('Time [h]', fontsize=16)
    consumptionfig.set(xticks=bins)

    #Making the figure pretty
    consumptionfig.tick_params(axis="both", which="both", bottom="off",
                               top="off", labelbottom="on", left="off",
                               right="off", labelleft="on")

    consumptionfig.set_axisbelow(True)
    consumptionfig.grid(b=True, which='major', axis='y', color='#cccccc',linestyle='--')

    if price is not None:
        pricefig = consumptionfig.twinx()
        pricefig.step(bins, price, color='black', where='mid', label='Price scheme')
        pricefig.set_ylabel('Price [NOK/kWh]', fontsize=16)
        consumptionfig.set_axisbelow(True)

    # Make the legend without border, and on the right side of the plot
    handles, labels = consumptionfig.get_legend_handles_labels()
    handle, label =pricefig.get_legend_handles_labels()
    consumptionfig.legend(bbox_to_anchor=(1.125, 1), loc=2, frameon=False,  fontsize=15)
    pricefig.legend(bbox_to_anchor=(1.125, 0), loc=2, frameon=False, fontsize=15)
    #consumptionfig.legend(bbox_to_anchor=(0.5, -0.35), loc=8, borderaxespad=0., fontsize=15)

    plt.tight_layout()
    plt.show()




"""
# Fra torsdag
def consumption_plot(shift=None, nonshift=None, shiftnames=None, nonshiftnames=None, price=None):
    """
    Generate a histogram plotof the appliance consumption, including
    a graphical line of the pricing scheme
    """
    f, consumptionfig = plt.subplots(1, 1, figsize=(10,6))

    if shift is not None:
        length = len(shift[0])
    elif nonshift is not 0:
        length = len(nonshift[0])
    elif price is not None:
        length = len(price)

    bins    = np.arange(0, length)
    width   = 0.9
    bottom  = np.zeros(length)

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
    consumptionfig.set_title('Household consumption ', fontweight='bold', size=16)
    consumptionfig.set_ylabel('Consumption [kWh]', fontsize=16)
    consumptionfig.set_xlabel('Time [h]', fontsize=16)
    consumptionfig.set(xticks=bins)

    #Making the figure pretty
    consumptionfig.tick_params(axis="both", which="both", bottom="off",
                               top="off", labelbottom="on", left="off",
                               right="off", labelleft="on")

    consumptionfig.set_axisbelow(True)
    consumptionfig.grid(b=True, which='major', axis='y', color='#cccccc',linestyle='--')

    if price is not None:
        pricefig = consumptionfig.twinx()
        pricefig.step(bins, price, color='black', where='mid', label='Price scheme')
        pricefig.set_ylabel('Price [NOK/kWh]', fontsize=16)
        consumptionfig.set_axisbelow(True)

    # Make the legend without border, and on the right side of the plot
    handles, labels = consumptionfig.get_legend_handles_labels()
    handle, label =pricefig.get_legend_handles_labels()
    consumptionfig.legend(bbox_to_anchor=(1.125, 1), loc=2, frameon=False,  fontsize=15)
    pricefig.legend(bbox_to_anchor=(1.125, 0), loc=2, frameon=False, fontsize=15)
    #consumptionfig.legend(bbox_to_anchor=(0.5, -0.35), loc=8, borderaxespad=0., fontsize=15)

    plt.tight_layout()
    plt.show()
"""
