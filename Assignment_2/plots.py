"""
IN5410 - Energy informatics | Assignment 2
"""
import datetime
import matplotlib.pyplot 		as plt
import matplotlib.dates         as mdates
import MachineLearning          as ML
import seaborn                  as sns
import Data                     as Data

from matplotlib.ticker          import MaxNLocator
from prettytable                import PrettyTable

# -----------------------------------------------------------------------------

def prediction_solution_plot(y_pred, power_solution, date, title="", figname='', savefig=False):
    """
    Function thats plots the predicted power vs. the actual generated power
    """
    fig, ax = plt.subplots(figsize=(8.8, 4.2))
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
        plt.savefig(figname);  print("--> Figure saved"); plt.show()
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
        plt.savefig(figname); print("--> Figure saved"); plt.show()
    else:
        plt.show()

def prediction_solution_plot_T3_1(y_pred, y_pred_svr, power_solution, date, title="", figname='', savefig=False):
    """
    Function thats plots the predicted power vs. the actual generated power
    """
    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    ax.plot_date(date, power_solution, 'g-', linewidth=0.9, label="Real")
    ax.plot_date(date, y_pred, 'b-', linewidth=0.9, label="Predicted, LR")
    ax.plot_date(date, y_pred_svr, 'm-', linewidth=0.9, label="Predicted, SVR")
    fig.autofmt_xdate(ha='center')
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
    ax.xaxis.set_major_locator(MaxNLocator(nbins='auto'))

    plt.title(title, fontsize=15)
    plt.xlabel("Year %s" %date.year[0], fontsize=15)
    plt.ylabel("Wind power [normalized]", fontsize=15)
    plt.legend(loc='lower right'); plt.grid(alpha=0.6, linewidth=0.5); plt.tight_layout()

    if savefig:
        plt.savefig(figname); print("--> Figure saved"); plt.show()
    else:
        plt.show()

def prediction_solution_plot_T3(y_pred, power_solution, title="", figname='', savefig=False):
    """
    Function thats plots the predicted power vs. the actual generated power
    """
    fig, ax = plt.subplots(figsize=(8.8, 4.2))
    ax.plot(power_solution, 'g-', linewidth=0.9, label="Real")
    ax.plot(y_pred, 'b-', linewidth=0.9, label="Predicted")

    plt.title(title, fontsize=15)
    plt.xlabel("Time", fontsize=15)
    plt.ylabel("Wind power [normalized]", fontsize=15)
    plt.legend(loc='lower right'); plt.grid(alpha=0.6, linewidth=0.5); plt.tight_layout()

    if savefig:
        plt.savefig(figname);  print("--> Figure saved"); plt.show()
    else:
        plt.show()

def Metrics(power_solution, y_pred, param="", method="", filename=""):

    x = PrettyTable()
    x.field_names = [param, "MSE", "RMSE", "R2"]

    x.add_row([method,  "%.3f"% ML.MSE(power_solution, y_pred), "%.3f"% ML.RMSE(power_solution, y_pred), "%.3f"% ML.R2(power_solution, y_pred)])
    #x.add_row(["RMSE", "%.3f"% ML.RMSE(power_solution, y_pred)])
    #x.add_row(["R2",   "%.3f"% ML.R2(power_solution, y_pred)])

    print(x)
    with open(filename, 'w') as w:
        w.write(str(x))

def Metrics_compare(power_solution, y_pred_lr, y_pred_mlr, filename=""):

    x = PrettyTable()
    x.field_names = ["", "Linear Regression", "Multiple Linear Regression"]
    x.align[""] = "l"

    x.add_row(["MSE",  "%.3f"% ML.MSE(power_solution, y_pred_lr),  "%.3f"% ML.MSE(power_solution, y_pred_mlr)])
    x.add_row(["RMSE", "%.3f"% ML.RMSE(power_solution, y_pred_lr), "%.3f"% ML.RMSE(power_solution, y_pred_mlr)])
    x.add_row(["R2",   "%.3f"% ML.R2(power_solution, y_pred_lr),   "%.3f"% ML.R2(power_solution, y_pred_mlr)])

    print(x)
    with open(filename, 'w') as w:
        w.write(str(x))


def Heatmap_MSE_R2(MSE, RMSE, R2, lmbd_vals, eta_vals, title='', figname='', savefigs=False):
    #seaborn.set(context='notebook', style='darkgrid', palette='deep', font='sans-serif', font_scale=1, color_codes=True, rc=None)

    # matplotlib 3.1.1 broke seaborn.... run: pip install matplotlib==3.1.2

    # Creating a heatmap of MSE values
    fig, ax = plt.subplots()
    sns.heatmap(MSE, annot=True, linewidths=.3, linecolor="black", ax=ax)

    ax.set_xticklabels(lmbd_vals, minor=False, ha='center')
    ax.set_yticklabels(eta_vals,  minor=False, va='center')

    plt.title(title + ' (MSE)', fontsize=15)
    plt.ylabel("$\\eta$",       fontsize=15)
    plt.xlabel("$\\lambda$",    fontsize=15)

    if savefigs:
        plt.savefig(figname + '_MSE')


    # Creating a heatmap of RMSE values
    fig, ax = plt.subplots()
    sns.heatmap(RMSE, annot=True, linewidths=.3, linecolor="black", ax=ax)

    ax.set_xticklabels(lmbd_vals, minor=False, ha='center')
    ax.set_yticklabels(eta_vals,  minor=False, va='center')

    plt.title(title + ' (RMSE)', fontsize=15)
    plt.ylabel("$\\eta$",        fontsize=15)
    plt.xlabel("$\\lambda$",     fontsize=15)

    if savefigs:
        plt.savefig(figname + '_RMSE')


    # Creating a heatmap of R2 values
    fig, ax = plt.subplots()
    sns.heatmap(R2, annot=True, linewidths=.3, linecolor="black", ax=ax)

    ax.set_xticklabels(lmbd_vals, minor=False, ha='center')
    ax.set_yticklabels(eta_vals,  minor=False, va='center')

    plt.title(title + ' (R2)', fontsize=15)
    plt.ylabel("$\\eta$",      fontsize=15)
    plt.xlabel("$\\lambda$",   fontsize=15)

    if savefigs:
        plt.savefig(figname + '_R2')

    #else:
    plt.show()
