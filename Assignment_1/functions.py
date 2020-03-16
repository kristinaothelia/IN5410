# IN5410 - Assignment 1: Functions
# -----------------------------------------------------------------------------
import os, random, xlsxwriter, sys, random

import pandas               as pd
import numpy                as np

from scipy.optimize         import linprog
from numpy.random           import RandomState
# -----------------------------------------------------------------------------

seed = 100
#seed = 3210
rs   = RandomState(seed)

def Get_df(file_name='/energy_use.xlsx'):
    """
    Function for reading the xlxs-file
    """

    cwd      = os.getcwd()
    filename = cwd + file_name
    nanDict  = {}
    df       = pd.read_excel(filename, header=0, skiprows=0, index_col=0, na_values=nanDict)

    #header_names = list(df)
    #print(df['Alpha']['EV'])
    return df

def applications(df):
    """
    Function that returns appliances data
    """
    n_app         = len(df)                     # Number of appliances
    app_names     = df.index.values

    # Get variables from the Excel file
    shiftable     = df[df['Shiftable'] == 1]    # Shiftable appliances
    non_shiftable = df[df['Shiftable'] == 0]    # Non-shiftable appliances

    alpha         = df['Alpha'].values          # Lower bounce. Set-up time
    beta          = df['Beta'].values           # Upper bounce. Deadline
    length        = df['Length [h]'].values     # Length: Power use per day [h]

    return n_app, app_names, shiftable, non_shiftable, alpha, beta, length


# OBSOBSOBS !!! Denne er ikke ferdig. Task 3
def applications_Task3(df, households):
    """
    Function that returns appliances data

    Some of the n-households have EVs.
    The last 6 shiftable appliances are randomly selected for the households
    At least 4 of these in each household.
    """

    # Disse stemmer ikke lengre, siden noe fjernes...
    #n_app         = len(df)                     # Number of appliances
    #app_names     = df.index.values

    # Get variables from the Excel file
    non_shiftable = df[df['Shiftable'] == 0]    # Non-shiftable appliances
    shiftable_set = df[df['Shiftable'] == 1]    # Shiftable appliances
    shiftable_ran = df[df['Shiftable'] == 2]    # Shiftable appliances - random

    non_shiftable_names = non_shiftable.index.values

    # Make sure only a fraction of the households have an EV
    #print(shiftable_set[shiftable_set.index == 'EV'])

    EV = random.randint(0, 1)
    if EV == 0:
        shiftable_set = shiftable_set[:-1]
    shiftable_set_names = shiftable_set.index.values

    # Now make a random selection of optional shiftable appliances,
    # where a household has 2-6 appliances
    optional = []
    while len(optional) < 2:
        random_appliances(shiftable_ran, optional)

    shiftable_ran_ = pd.DataFrame(optional)
    shiftable_ran_names = shiftable_ran_.index.values

    # Number of appliances
    n_app    = len(non_shiftable) + len(shiftable_set) + len(shiftable_ran_)

    alpha    = non_shiftable['Alpha'].values        # Lower bounce. Set-up time
    alpha_s  = shiftable_set['Alpha'].values
    alpha_r  = shiftable_ran['Alpha'].values
    #print(alpha)
    #print(alpha_s)
    #print(alpha_r)
    beta     = non_shiftable['Beta'].values         # Upper bounce. Deadline
    beta_s   = shiftable_set['Beta'].values
    beta_r   = shiftable_ran['Beta'].values

    length   = non_shiftable['Length [h]'].values   # Power use per day [h]
    length_s = shiftable_set['Length [h]'].values
    length_r = shiftable_ran['Length [h]'].values

    # Test:
    alpha_combined = []
    alpha_combined.append(alpha)
    alpha_combined.append(alpha_s)
    alpha_combined.append(alpha_r)
    print(list(alpha_combined))         # æææææææææææææææææææææææææææ!!!


    # Maa returnere kombinerte alpha, beta, length lister
    # Maa ogsaa returnere kombinerte pandas frames for appliances?
    return  n_app, alpha, alpha_s, alpha_r, beta, beta_s, beta_r, \
            length, length_s, length_r, non_shiftable, non_shiftable_names, \
            shiftable_set, shiftable_set_names, shiftable_ran_, shiftable_ran_names


def random_appliances(shiftable_ran, optional):

    for i in range(len(shiftable_ran)):
        # make random numbers
        n = random.randint(0, 1)
        # Add random appliances to the household
        if n == 1:
            optional.append(shiftable_ran.iloc[i])


def Get_price(hours, seed, ToU=False):
    """
    Function returning either Time of Use (ToU) or Real Time Price (RTP)
    """

    if ToU == True:                             # ToU price
        price        = [0.5] * hours            # General energy price
        price[17:20] = [1.0, 1.0, 1.0]          # After-work peak

    else:                                       # RTP price
        price        = [0.5] * hours            # General energy price
        price[6:9]   = [0.65, 0.75, 0.65]       # Morning peak
        price[16:20] = [0.65, 1.0, 1.0, 1.0]    # After-work peak

        price = np.array(price)

        for i in range(len(price)):
            # Outside peak-hour. Small variations during the night
            if i < 6:
                price[i] = price[i] + rs.uniform(-0.1,0.1)
            elif i < 17 or i > 20:
                price[i] = price[i] + rs.uniform(-0.2,0.2)
            # During peak-hour, some higher random variation
            else:
                price[i] = price[i] + rs.uniform(-0.2,0.4)

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

    # Set interval hours to 1, meaning that these hour are availiable for energy use.
    # Example: EV charging, 3h. linprog will deside which 3 of 6 h the EV will charge based on minimizing c.
    interval[:] = 1

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

    energy_hour = df['Hourly usage [kW]'].values    # Power Use [kW]
    E_tot       = [np.sum(energy_hour)]*hours


    A_eq = np.zeros((n_app, n_app*hours))
    b_eq = df['Daily usage [kWh]'].values

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
