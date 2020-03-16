# IN5410 - Assignment 1: Plotting
# -----------------------------------------------------------------------------
import os, random, xlsxwriter, sys, random

import matplotlib.pyplot    as plt
import numpy                as np

from datetime               import time
# -----------------------------------------------------------------------------

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
        plt.savefig("Figures/Task1_hist.png")
    else:
        plt.title("Time of Use [RTP]", fontsize='16', weight='bold')
        plt.savefig("Figures/Task2_hist.png")
    #plt.show()


def consumption_plot(price, app=None, non_app=None, app_names=None, non_app_names=None):
    """
    Generate a histogram plot of the appliance consumption, including a
    graphical line of the pricing scheme

    not None => True. None means none, not zero or False

    price           | c
    app             | Shiftable appliances
    app_names       | Shiftable appliances - names
    non_app         | Non-shiftable appliances
    non_app_names   | Non-shiftable appliances - names
    """

    fig, ax = plt.subplots(1, 1, figsize=(10,6))    # ax = consumption fig.
    length  = len(price)
    """
    if app is not None:
        length = len(app[0]) # Length of the first appliance
    elif non_app is not None:
        length = len(non_app[0])
    else: # elif price is not None:
        length = len(price)
    """
    bins    = np.arange(0, length)
    width   = 0.9
    bottom  = np.zeros(length)      # HVA BRUKES DENNE TIL?

    # Iterate over (shiftable) appliances to create stacked bars for the hist.
    if app is not None:
        for i in range(len(app)):

            ax.bar(bins, app[i], width=width, bottom=bottom, label=app_names[i])
            bottom = np.add(bottom, app[i])

    # Iterate over (non-shift.) appliances to create stacked bars for the hist.
    if non_app is not None:     # is not 0
        for i in range(len(non_app)):

            ax.bar(bins, non_app[i], width=width, bottom=bottom, label=non_app_names[i])
            bottom = np.add(bottom, non_app[i])

    # Set title and x/y-label
    ax.set_title('Household consumption ', fontweight='bold', size=16)
    ax.set_ylabel('Consumption [kWh]', fontsize=16)
    ax.set_xlabel('Time [h]', fontsize=16)
    ax.set(xticks=bins)

    # Place axis and make grid.
    ax.set_axisbelow(True)
    ax.grid(b=True, which='major', axis='y', color='#cccccc', linestyle='--')

    # Make a line that reprecents the hourly pricing scheme, p_line
    if price is not None:
        p_line = ax.twinx()       # Create a twin axes sharing the x-axis
        p_line.step(bins, price, color='black', where='mid', label='Price scheme')
        p_line.set_ylabel('Price [NOK/kWh]', fontsize=16)
        ax.set_axisbelow(True)

    # Make the legend without border, and on the right side of the plot
    handles_con, labels_con     = ax.get_legend_handles_labels()
    handles_price, labels_price = p_line.get_legend_handles_labels()

    ax.legend(bbox_to_anchor=(1.125, 1),     loc=2, frameon=False, fontsize=15)
    p_line.legend(bbox_to_anchor=(1.125, 0), loc=2, frameon=False, fontsize=15)

    plt.tight_layout()
    #plt.show()


"""
# Fra torsdag
def consumption_plot(shift=None, nonshift=None, shiftnames=None, nonshiftnames=None, price=None):

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
