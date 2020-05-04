"""
IN5410 - Energy informatics | Assignment 2
"""

import matplotlib.pyplot 		as plt
import matplotlib.dates         as mdates

from matplotlib.ticker import MaxNLocator
# -----------------------------------------------------------------------------

def prediction_solution_plot(y_pred, power_solution, date, title=""):
    """
    Function thats plots the predicted power vs. the actual generated power
    """
    fig, ax = plt.subplots(figsize=(8.5, 4.5)) # dpi=80
    ax.plot_date(date, y_pred, fmt='-', label="Predicted")
    ax.plot_date(date, power_solution, fmt='-', label="Real")
    fig.autofmt_xdate(ha='center')
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
    ax.xaxis.set_major_locator(MaxNLocator(nbins='auto'))
    plt.title(title, fontsize=15)
    plt.xlabel("Year %s" %date.year[0], fontsize=15)
    plt.ylabel("Power [???]", fontsize=15)
    plt.legend();	plt.tight_layout(); 	plt.grid()
    plt.show()