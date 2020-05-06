"""
IN5410 - Energy informatics | Assignment 2
"""
import datetime
import matplotlib.pyplot 		as plt
import matplotlib.dates         as mdates
import readData                 as Data

from matplotlib.ticker          import MaxNLocator

# -----------------------------------------------------------------------------

def prediction_solution_plot(y_pred, power_solution, date, title="", figname='', savefig=False):
    """
    Function thats plots the predicted power vs. the actual generated power
    """
    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    ax.plot_date(date, power_solution, 'g-', linewidth=0.9, label="Real")
    ax.plot_date(date, y_pred, 'b-', linewidth=0.9, label="Predicted")
    fig.autofmt_xdate(ha='center')
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
    ax.xaxis.set_major_locator(MaxNLocator(nbins='auto'))

    plt.title(title, fontsize=15)
    plt.xlabel("Year %s" %date.year[0], fontsize=15)
    plt.ylabel("Wind power [normalized]", fontsize=15)
    plt.legend(loc='lower right'); plt.grid(alpha=0.6, linewidth=0.5); plt.tight_layout()

    if savefig:
        plt.savefig(figname)
    else:
        plt.show()

def prediction_solution_plot_T2(y_pred, y_pred_mlr, power_solution, date, title="", figname='', savefig=False):
    """
    Function thats plots the predicted power vs. the actual generated power
    """
    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    ax.plot_date(date, power_solution, 'g-', linewidth=0.9, label="Real")
    ax.plot_date(date, y_pred, 'b-', linewidth=0.9, label="Predicted, LR")
    ax.plot_date(date, y_pred_mlr, 'm-', linewidth=0.9, label="Predicted, MLR")
    fig.autofmt_xdate(ha='center')
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
    ax.xaxis.set_major_locator(MaxNLocator(nbins='auto'))

    plt.title(title, fontsize=15)
    plt.xlabel("Year %s" %date.year[0], fontsize=15)
    plt.ylabel("Wind power [normalized]", fontsize=15)
    plt.legend(loc='lower right'); plt.grid(alpha=0.6, linewidth=0.5); plt.tight_layout()

    if savefig:
        plt.savefig(figname)
    else:
        plt.show()
