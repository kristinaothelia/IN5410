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
        plt.title("Real Time Pricing [RTP]", fontsize='16', weight='bold')
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

    fig, ax = plt.subplots(1, 1, figsize=(9,7))    # ax = consumption fig.
    #fig, ax = plt.subplots(1, 1, figsize=(10,6))    # ax = consumption fig.
    length  = len(price)

    bins    = np.arange(0, length)
    width   = 0.9
    bottom  = np.zeros(length)

    #cmap = plt.get_cmap('hsv')
    #colors = [cmap(i) for i in np.linspace(0, 1, len(app)+1)]

    colors = ['firebrick','springgreen','yellow','slategray','magenta','khaki','orangered','slateblue','blue','lime','purple','green','red','saddlebrown','darkturquoise','black']
    Tot = 0
    # Iterate over shiftable appliances to create stacked bars for the hist.
    if app is not None:
        for i in range(len(app)):

            ax.bar(bins, app[i], color=colors[i], width=width, bottom=bottom, label=app_names[i])
            bottom = np.add(bottom, app[i])
            Tot += app[i]

    # Iterate over non-shift. appliances to create stacked bars for the hist.
    if non_app is not None:
        for i in range(len(non_app)):

            ax.bar(bins, non_app[i], width=width, bottom=bottom, label=non_app_names[i])
            bottom = np.add(bottom, non_app[i])
            Tot += non_app[i]

    # Make a max power load line
    plt.axhline(y=max(Tot), color='r', linestyle='--', label='Max power load = %0.1f kW' % max(Tot))

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

    """
    # Make the legend without border, and on the right side of the plot
    handles_con, labels_con     = ax.get_legend_handles_labels()
    handles_price, labels_price = p_line.get_legend_handles_labels()

    ax.legend(bbox_to_anchor=(1.125, 1),     loc=2, frameon=False, fontsize=15)
    p_line.legend(bbox_to_anchor=(1.125, 0), loc=2, frameon=False, fontsize=15)
    """

    #handles_con, labels_con     = ax.get_legend_handles_labels()
    #handles_price, labels_price = p_line.get_legend_handles_labels()

    #ax.legend(loc='upper center', bbox_to_anchor=(0.5,-0.1), frameon=False, fontsize=15, ncol=3)
    p_line.legend(loc='upper center', bbox_to_anchor=(0.884, 1.2), frameon=False, fontsize=13)
    #p_line.legend(loc='upper left', frameon=False, fontsize=13)
    #p_line.legend(loc='best', frameon=False, fontsize=13)
    ax.legend(loc='lower left', bbox_to_anchor= (0.0, 1.1), ncol=3, borderaxespad=0, frameon=False, fontsize=13)

    #text = ax.text(-0.2,1.05, "Aribitrary text", transform=ax.transAxes)
    #lgd = ax.legend(handles_con, labels_con, loc='upper center', bbox_to_anchor=(0.5,-0.1))

    plt.tight_layout()
    #fig.savefig('samplefigure', bbox_extra_artists=(lgd,text), bbox_inches='tight', ncol=3)
    #plt.show()



def consumption_plot_Task3(price, EV, app=None, non_app=None, app_names=None, non_app_names=None):
    """
    Generate a histogram plot of the appliance consumption, including a
    graphical line of the pricing scheme

    This function only plotts non-shiftable and shiftable consumption!

    price           | c
    app             | Shiftable appliances
    app_names       | Shiftable appliances - names
    non_app         | Non-shiftable appliances
    non_app_names   | Non-shiftable appliances - names
    """

    #fig, ax = plt.subplots(1, 1, figsize=(10,6))    # ax = consumption fig.
    fig, ax = plt.subplots(1, 1, figsize=(9,6))    # ax = consumption fig.
    length  = len(price)

    bins    = np.arange(0, length)
    width   = 0.9
    bottom  = np.zeros(length)

    # Plot histogram/bars for non-shiftable and shiftable app.
    if non_app is not None:
        ax.bar(bins, non_app, width=width, bottom=bottom, label=non_app_names)
        bottom = np.add(bottom, non_app)

    if app is not None:
        ax.bar(bins, app, width=width, bottom=bottom, label=app_names)
        bottom = np.add(bottom, app)

    # Set title and x/y-label
    ax.set_title('Neighborhood consumption (%g EVs)' %EV, fontweight='bold', size=16)
    ax.set_ylabel('Consumption [kWh]', fontsize=16)
    ax.set_xlabel('Time [h]', fontsize=16)
    ax.set(xticks=bins)

    # Make a max power load line
    Tot = app + non_app
    plt.axhline(y=max(Tot), color='r', linestyle='--', label='Max power load = %0.1f kW' % max(Tot))


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
    #handles_con, labels_con     = ax.get_legend_handles_labels()
    #handles_price, labels_price = p_line.get_legend_handles_labels()

    ax.legend(loc='lower center', bbox_to_anchor= (0.5, 1.1), ncol=2, borderaxespad=0, frameon=False, fontsize=13)
    p_line.legend(loc='lower center', bbox_to_anchor=(0.8, -0.175), ncol=2, frameon=False, fontsize=13)

    plt.tight_layout()
