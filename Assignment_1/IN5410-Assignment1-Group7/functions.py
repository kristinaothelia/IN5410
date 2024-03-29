# IN5410 - Assignment 1: Functions
# -----------------------------------------------------------------------------
import os, random, xlsxwriter, sys, random, excel2img

import pandas               as pd
import numpy                as np

from scipy.optimize         import linprog
from numpy.random           import RandomState
# -----------------------------------------------------------------------------

#seed = 100
seed = 3210
rs   = RandomState(seed)

def Get_df(file_name='/energy_use.xlsx'):
    """
    Function for reading the xlxs-file
    """

    cwd      = os.getcwd()
    filename = cwd + file_name
    nanDict  = {}
    df       = pd.read_excel(filename, header=0, skiprows=0, index_col=0, na_values=nanDict)

    return df

def applications(df, households=0, Task3=False):
    """
    Function that returns appliances data

    For Task 3:
    Some of the n-households have EVs.
    The last 6 shiftable appliances are randomly selected for the households
    At least 2 of these in each household.
    """

    if Task3:

        # Get variables from the Excel file
        non_shiftable = df[df['Shiftable'] == 0]    # Non-shiftable appliances
        shiftable_set = df[df['Shiftable'] == 1]    # Shiftable appliances
        shiftable_ran = df[df['Shiftable'] == 2]    # Shiftable appliances - random

        non_shiftable_names = non_shiftable.index.values

        # Make sure only a fraction of the households have an EV
        EV = random.randint(0, 1)
        EV_nr = 0
        if EV == 0:
            # Dette vil ikke funke om excel filen endres.
            # Burde bruke shiftable_set[shiftable_set.index == 'EV'] greier...
            shiftable_set = shiftable_set[:-1]
        else:
            EV_nr = 1

        shiftable_set_names = shiftable_set.index.values

        # Now make a random selection of optional shiftable appliances,
        # where a household has 2-6 appliances
        optional = []
        while len(optional) < 2:
            optional = []
            random_appliances(shiftable_ran, optional)

        shiftable_ran_      = pd.DataFrame(optional)
        shiftable_ran_names = shiftable_ran_.index.values

        shiftable_combined  = pd.concat([shiftable_set, shiftable_ran_]) #, axis=0
        shiftable_c_names   = shiftable_combined.index.values

        # Number of appliances
        n_app    = len(non_shiftable) + len(shiftable_combined)

        alpha    = non_shiftable['Alpha'].values        # Lower bounce. Set-up time
        alpha_c  = shiftable_combined['Alpha'].values

        beta     = non_shiftable['Beta'].values         # Upper bounce. Deadline
        beta_c   = shiftable_combined['Beta'].values

        length   = non_shiftable['Length [h]'].values   # Power use per day [h]
        length_c = shiftable_combined['Length [h]'].values

        alpha    = non_shiftable['Alpha'].values        # Lower bounce. Set-up time
        alpha_s  = shiftable_set['Alpha'].values

        alpha_combined     = np.concatenate([alpha, alpha_c]).astype(int)
        beta_combined      = np.concatenate([beta, beta_c]).astype(int)
        length_combined    = np.concatenate([length, length_c]).astype(int)

        N_ = shiftable_ran_names
        return  n_app, alpha_combined, beta_combined, length_combined, \
                non_shiftable, non_shiftable_names, shiftable_combined, \
                shiftable_c_names, EV_nr, N_

    else:
        # For Task 1 and Task 2
        n_app         = len(df)                     # Number of appliances
        app_names     = df.index.values

        alpha         = df['Alpha'].values          # Lower bounce. Set-up time
        beta          = df['Beta'].values           # Upper bounce. Deadline
        length        = df['Length [h]'].values     # Length: Power use per day [h]

        return n_app, app_names, alpha, beta, length

def calc_households(nr_non_shiftable, households=30, hours=24, make_result_table=True):
    """
    Makes calculations simular to task2, bur for 30 households.
    Sums up the total cost and number of EVs for the neighborhood
    """

    # Fill up arrays with total consumption for all households
    Total_con_n = np.zeros(hours)       # Non-shiftable
    Total_con_s = np.zeros(hours)       # Shiftable

    # Get pricing scheme. ToU (Time-of-Use) or RTP (Real-Time-Pricing)
    price = Get_price(hours, seed=seed, ToU=False)
    cost  = 0

    # Skal alle husholdningene ha samme pris, eller skal dette genereres ulikt?
    EV_number    = 0
    house_nr     = []
    cost_nr      = []
    hav_nonshift = []
    hav_shift    = []
    hav_tot      = []
    EV_yes_no    = []
    Names        = [] # name of random shiftable appliances

    for i in range(households):
        df    = Get_df(file_name='/energy_use.xlsx')   # Get data for appliances

        n_app, alpha, beta, length, non_shiftable, non_shiftable_names, \
        shiftable_combined, shiftable_c_names, EV_nr, N_ \
        = applications(df, households, Task3=True)

        # Make shorter names for the output (LaTeX) table
        name_1 = ''
        for navn in range(len(N_)):

            if N_[navn] == 'Coffee maker ':
                name_1 +=  'CM, '
            elif N_[navn] == 'Microwave':
                name_1 +=  'MW, '
            elif N_[navn] == 'Cellphone charger':
                name_1 +=  'CC, '
            elif N_[navn] == 'Hair dryer':
                name_1 +=  'HD, '
            elif N_[navn] == 'Game console':
                name_1 +=  'GC, '
            elif N_[navn] == 'Wi-Fi router':
                name_1 +=  'WiFi, '
            else:
                name_1 += N_[navn] + ', '
        Names.append(name_1[:-2])

        if EV_nr == 1:
            EV_number += 1
            EV_yes_no.append('Yes')
        else:
            EV_yes_no.append('No')

        # Creating intervals
        intervals = interval_severeal_app(n_app, length, alpha, beta)

        df = pd.concat((non_shiftable, shiftable_combined))

        # Make vriables for linprog. c, A_eq, b_eq, A_ub, b_ub
        c, A_eq, b_eq, A_ub, b_ub = linprog_input(df, n_app, price, intervals, hours)

        # Make linprog calculations
        res         = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=(0,None))
        consumption = res.x.reshape(n_app, hours)

        non_s_con = consumption[:nr_non_shiftable]
        shift_con = consumption[nr_non_shiftable:]

        non_shift_tot = np.sum(non_s_con, axis=0)
        #print('Total hourly consumption for non-shiftable app.', '\n', non_shift_tot)
        shift_tot    = np.sum(shift_con, axis=0)
        #print('Total hourly consumption for shiftable app.', '\n', shift_tot)

        hav_nonshift.append(np.sum(non_shift_tot))
        hav_shift.append(np.sum(shift_tot))
        hav_tot.append(np.sum(non_shift_tot)+np.sum(shift_tot))

        Total_con_n  += non_shift_tot
        Total_con_s  += shift_tot

        cost += res.fun

        if i < 9:
            print("House %g,  Minimized cost: %.1f NOK" % (i+1, res.fun))
        else:
            print("House %g, Minimized cost: %.1f NOK" % (i+1, res.fun))

        #house_nr.append('House ' + '%g' %(i+1))
        house_nr.append(i+1)
        cost_nr.append('%.1f' %res.fun)

    if make_result_table == True:
        result_table(hav_nonshift, hav_shift, hav_tot, cost_nr, Names, EV_yes_no, house_nr)


    return df, price, EV_number, Total_con_s, Total_con_n, cost

def random_appliances(shiftable_ran, optional):
    """
    Function that randomly adds appliances to household
    """

    for i in range(len(shiftable_ran)):
        # make random numbers
        n = random.randint(0, 1)
        # Add random appliances to the household
        if n == 1:
            optional.append(shiftable_ran.iloc[i])

def Get_price(hours, seed, ToU=False):
    """
    Function returning either Time of Use (ToU) or Real Time Price (RTP).
    The price is returned as a list with length hours
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
            # During peak-hour, some higher random variation
            elif i >= 6 and i < 9:
                price[i] = price[i] + rs.uniform(-0.2,0.4)
            # During peak-hour, some higher random variation
            elif i >= 16 and i < 20:
                price[i] = price[i] + rs.uniform(-0.2,0.4)
            # Outside peak-hour. Small variations during the day
            else:
                price[i] = price[i] + rs.uniform(-0.15,0.2)

    return list(price)

def interval(hour=1, alpha=0, beta=23):
    """
    Creating a time interval, returned as 'numpy.ndarray'

    alpha       | Setup Time
    beta        | Deadline Time
    hour        | Amount of time the app have to be active
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

def interval_severeal_app(n_app, length, alpha, beta):
    """
    Make intervals for several appliances
    """
    intervals = []
    for i in range(n_app):
        intervals.append(interval(hour=length[i], alpha=alpha[i], beta=beta[i]))

    return np.array(intervals)

def linprog_input(df, n_app, price, intervals, hours=24):
    """
    Function returning the inputs needed in linprog,
    all the parameters are returned as 'numpy.ndarray', dtype=float64

    c       | The coefficients of the linear objective function to be minimized.
              Pricing scheme. Shape: (n_app*24,)

    A_eq    | The equality constraint matrix.
              Matrix with pre-filled time intervals for all appliactions

              Ex if we have 2 applications, with a 24-hour interval.
              1's is within the time boundaries, where the app. can run.
              Shape/Output:
              [[0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 1. 1. 1. 1.
                0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
               [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.
                0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 1. 1. 1. 1. 1. 1. 0.]]

    b_eq    | The equality constraint vector, Daily usage [kW].
              Daily Use [kWh] for all appliances. Shape: (n_app,)
              --> [Daily use(app1), Daily use(app2), ...]

    A_ub    | The inequality constraint matrix.
              Consists of two matrices concatenated together, A_ub.
              The first matrix ,A_, has shape (24, 24*n_app)
              Where we add a '1' in each row, one column at a time.
              (Like a identity matrix with truncated representation)
              The next, A_id, is an identity matrix.
              Shape: (n_app*24, n_app*24)

    b_ub    | The inequality constraint vector.
              Power use for all appliances. Shape: (n_app*24+1*24,)
              --> [Power use(app1)*24 + Power use(app2)*24 +...+ (Tot_energy for all apps)*24]
    """

    c    = np.array(price*len(df))
    b_eq = df['Daily usage [kWh]'].values # Total daily usage of the appliances

    A_eq = np.zeros((n_app, n_app*hours))
    for i in range(n_app):
        for j in range(hours):
            A_eq[i,j+(hours*i)] = intervals[i][j]

    np.set_printoptions(threshold=np.inf)
    A_ = np.zeros((hours,n_app*hours))
    for i in range(A_.shape[0]):
        A_[i,i::hours] = 1

    A_id  = np.eye(n_app*hours)
    A_ub  = np.concatenate((A_id,A_),axis=0)

    energy_hour = df['Hourly usage [kW]'].values  # Power Use [kW]
    E_tot       = [np.sum(energy_hour)]*hours
    b           = []
    for i in energy_hour:
        b.append([i]*hours)
    b    = np.array(b).ravel()
    b_ub = np.concatenate((b, E_tot))

    return c, A_eq, b_eq, A_ub, b_ub

def result_table(hav_nonshift, hav_shift, hav_tot, cost_nr, Names, EV_yes_no, house_nr):
    """
    A function which creates a Pandas DataFrame with the household's results,
    and export the DataFrame as an excel-file, a LaTex-table and a png-image:
    result_table.xlsx
    latex_table.tex
    result_figure.png
    """

    # Creates a pandas DataFrame (result_table) of the input lists
    list_of_tuples = list(zip(hav_nonshift, hav_shift, hav_tot, Names, cost_nr, EV_yes_no))
    result_table   = pd.DataFrame(list_of_tuples,index=house_nr,\
                         columns = ['Non-shiftable [kWh]', 'Shiftable [kWh]', 'Total [kWh]', \
                                    'Optional app.','Minimized cost [NOK]', 'EV'])

    l = len(result_table)         # length of table
    w = len(result_table.columns) # widt of table

    # Creates an excel-document of the dataframe 'result_table'
    writer   = pd.ExcelWriter('result_table.xlsx', engine='xlsxwriter')
    result_table.to_excel(writer, sheet_name='task3', float_format="%.3f", index_label='House')
    workbook = writer.book

    # Creating workbook format to center values
    format1     = workbook.add_format({'align': 'center','bold': False})
    text_format = workbook.add_format({'text_wrap': True})

    #Columns.AutoFit()

    # Creating worksheet to edit edit colums and add formats
    worksheet   = writer.sheets['task3']
    worksheet.set_column('B:D', None, format1)
    worksheet.set_column('F:G', None, format1)
    worksheet.set_column('B:B', 18, None)
    worksheet.set_column('C:C', 14, None)
    worksheet.set_column('D:D', 12, None)
    worksheet.set_column('E:E', 22, text_format)
    worksheet.set_column('F:F', 18, None)

    house_format  = workbook.add_format({'bottom':2, 'top':5, 'left':5, 'right':1, 'bg_color': '#C6EFCE'})
    header_format = workbook.add_format({'bottom':2, 'top':5, 'left':0, 'right':2, 'bg_color': '#C6EFCE'})
    worksheet.conditional_format(xlsxwriter.utility.xl_range(0, 0, l, 0), {'type': 'no_errors', 'format': house_format})
    worksheet.conditional_format(xlsxwriter.utility.xl_range(0, 0, 0, w), {'type': 'no_errors', 'format': header_format})

    writer.save() # saves the document

    # Creates a .tex table which can be imported or copied to a latex document
    latex_table = result_table.to_latex('latex_table.tex', float_format="%.2f")

    # Exports a png image of the result table
    excel2img.export_img('result_table.xlsx','result_figure.png')  # pip install excel2img
