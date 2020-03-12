import os, random, xlsxwriter, sys, argparse

import matplotlib.pyplot 	as plt
import numpy               	as np
import functions 		   	as func

from scipy.optimize 		import linprog
from random 				import seed
# Python 3.7.4
#------------------------------------------------------------------------------
seed = 5410

df 	  = func.Get_df(file_name='/energy_use.xlsx')	# Get data for appliances
#df 	  = df[-3:]  # Only look at the 3 last appliances

hours = 24

#------------------------------------------------------------------------------


if __name__ == '__main__':

	parser = argparse.ArgumentParser(description="Minimize energy cost")

	group = parser.add_mutually_exclusive_group()
	group.add_argument('-1', '--Task1', action="store_true", help="3 shiftable appliances")
	group.add_argument('-2', '--Task2', action="store_true", help="Shiftable and non-shiftable appliances")
	group.add_argument('-3', '--Task3', action="store_true", help="30 households")

	# Optional argument for plotting
	parser.add_argument('-X', '--plot', action='store_true', help="Plotting", required=False)

	if len(sys.argv) <= 1:
		sys.argv.append('--help')

	args   = parser.parse_args()

	Task1  = args.Task1
	Task2  = args.Task2
	Task3  = args.Task3
	Plot   = args.plot

	if Task1 == True:

		print("Task1")
		print("--"*55)

		df = df[-3:]     # Only look at the 3 last appliances

		n_app, app_names, shiftable, non_shiftable, alpha, beta, length = func.applications(df)

		# Get pricing scheme. ToU (Time-of-Use) or RTP (Real-Time-Pricing)
		price = func.Get_price(hours, seed=seed, ToU=True)

		# Creating intervals
		intervals = []
		for i in range(n_app):
			intervals.append(func.interval(hour=length[i], start=alpha[i], stop=beta[i], shuffle=False))

		intervals = np.array(intervals)

		#------------------------------------------------------------------------------
		# Make vriables for linprog. c, A_eq, b_eq, A_ub, b_ub
		c, A_eq, b_eq, A_ub, b_ub = func.linprog_input(df, n_app, price, intervals, hours)


		res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=(0,None))

		consumption = res.x.reshape(n_app,hours)


		if Plot == True:
			# Make histogram of pricing scheme
			func.Make_p_hist(price)
			func.consumption_plot(shift=consumption, price=price, nonshift=0, shiftnames=app_names)


		else:
			print(res)
			print(str(res.fun))

	elif Task2 == True:

		print("Task2")
		print("--"*55)

		df = df[-3:]     # Only look at the 3 last appliances

		n_app, app_names, shiftable, non_shiftable, alpha, beta, length = func.applications(df)

		# Get pricing scheme. ToU (Time-of-Use) or RTP (Real-Time-Pricing)
		price = func.Get_price(hours, seed=seed, ToU=False)

		# Creating intervals
		intervals = []
		for i in range(n_app):
			intervals.append(func.interval(hour=length[i], start=alpha[i], stop=beta[i], shuffle=False))

		intervals = np.array(intervals)

		#------------------------------------------------------------------------------
		# Make vriables for linprog. c, A_eq, b_eq, A_ub, b_ub
		c, A_eq, b_eq, A_ub, b_ub = func.linprog_input(df, n_app, price, intervals, hours)


		res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=(0,None))

		consumption = res.x.reshape(n_app,hours)


		if Plot == True:
			# Make histogram of pricing scheme
			func.Make_p_hist(price)
			func.consumption_plot(shift=consumption, price=price, nonshift=0, shiftnames=app_names)


		else:
			print(res)
			print(str(res.fun))


	elif Task3 == True:

		print("Task3")
		print("--"*55)



		
