"""
IN5410 - Energy informatics | Assignment 2
"""

import matplotlib.pyplot 		as plt

import readData                as Data 
import datetime
# -----------------------------------------------------------------------------

def prediction_solution_plot(y_pred, power_solution, title=""):
    # Lage egen plotte ting...

    #plt.style.use('dark_background')
    plt.figure(figsize=(8.5, 4.5)) # dpi=80
    plt.plot(y_pred, label="Predicted")
    plt.plot(power_solution, label="Real")
    plt.title(title)
    plt.xlabel("Time [???]")
    plt.ylabel("Power [???]")
    plt.legend();	plt.tight_layout(); 	plt.grid()
    plt.show()


Solution  = Data.Get_data(filename='/Solution.csv')

# Dato for plotting
timestamp = Solution.index
year  = [x[0:4] for x in timestamp]
month = [x[4:6] for x in timestamp]
date  = [x[6:8] for x in timestamp]

print(date)
times = datetime.date(year[1], month[1], date[1])

print(times)
