import os, random, xlsxwriter, sys, argparse

import matplotlib.pyplot 	as plt
import numpy               	as np
import functions 		   	as func
import plotting 			as P

from scipy.optimize 		import linprog
from random 				import seed

# Python 3.7.4
#------------------------------------------------------------------------------
seed  = 5410

df 	  = func.Get_df(file_name='/energy_use.xlsx')	# Get data for appliances
hours = 24


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

		print("--"*40); print("Task 1"); print("--"*40)
		# ---------------------------------------------------------------------

		# Only look at appliances: Dishwasher, LM and EV
		#df = df[-3:]
		df = df[7:10]

		n_app, app_names, shiftable, non_shiftable, alpha, beta, length = func.applications(df)

		# Get pricing scheme. ToU (Time-of-Use) or RTP (Real-Time-Pricing)
		price = func.Get_price(hours, seed=seed, ToU=True)

		# Creating intervals
		intervals = func.interval_severeal_app(n_app, length, alpha, beta, shuffle=False)

		# Make vriables for linprog. c, A_eq, b_eq, A_ub, b_ub
		c, A_eq, b_eq, A_ub, b_ub = func.linprog_input(df, n_app, price, intervals, hours)

		# Make linprog calculations
		res 		= linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=(0,None))
		consumption = res.x.reshape(n_app,hours)

		if Plot == True:
			P.Make_p_hist(df, price)
			P.consumption_plot(price=price, app=consumption, app_names=app_names)
			plt.show()

		else:
			#print(res)
			print(res.message)
			print("Status: ", res.status)
			print("Minimized cost: %.3f" % res.fun)


	elif Task2 == True:

		print("--"*40); print("Task 2"); print("--"*40)
		# ---------------------------------------------------------------------

		n_app, app_names, shiftable, non_shiftable, alpha, beta, length = func.applications(df)

		# Get pricing scheme. ToU (Time-of-Use) or RTP (Real-Time-Pricing)
		price = func.Get_price(hours, seed=seed, ToU=False)

		# Creating intervals
		intervals = func.interval_severeal_app(n_app, length, alpha, beta, shuffle=False)

		# Make vriables for linprog. c, A_eq, b_eq, A_ub, b_ub
		c, A_eq, b_eq, A_ub, b_ub = func.linprog_input(df, n_app, price, intervals, hours)

		# Make linprog calculations
		res 		= linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=(0,None))
		consumption = res.x.reshape(n_app,hours)

		if Plot == True:
			P.Make_p_hist(df, price)
			P.consumption_plot(price=price, app=consumption, app_names=app_names)
			#P.consumption_plot(shift=consumption, price=price, nonshift=0, shiftnames=app_names)
			plt.show()

		else:
			#print(res)
			print(res.message)
			print("Status: ", res.status)
			print("Minimized cost: %.3f" % res.fun)


	elif Task3 == True:

		print("--"*40); print("Task 3"); print("--"*40)
		# ---------------------------------------------------------------------
		households = 2

		# Get pricing scheme. ToU (Time-of-Use) or RTP (Real-Time-Pricing)
		price = func.Get_price(hours, seed=seed, ToU=False)

		for i in range(households):

			n_app, alpha, alpha_s, alpha_r, beta, beta_s, beta_r, \
		    length, length_s, length_r, non_shiftable, non_shiftable_names, \
		    shiftable_set, shiftable_set_names, shiftable_ran, shiftable_ran_names \
			= func.applications_Task3(df, households)

			print(n_app, len(non_shiftable), len(shiftable_set), len(shiftable_ran))

			"""
			# Creating intervals
			intervals = func.interval_severeal_app(n_app, length, alpha, beta, shuffle=False)

			# Make vriables for linprog. c, A_eq, b_eq, A_ub, b_ub
			c, A_eq, b_eq, A_ub, b_ub = func.linprog_input(df, n_app, price, intervals, hours)

			# Make linprog calculations
			res 		= linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=(0,None))
			consumption = res.x.reshape(n_app,hours)
			print("Minimized cost: %.3f" % res.fun)
			"""

		"""
		if Plot == True:

			# Only need to plot combined shiftable and non-shiftable?

			P.Make_p_hist(df, price)

			consumption_plot(price=price, app=, non_app=, app_names=, non_app_names=)

			#P.consumption_plot(price=price, app=consumption, app_names=app_names)
			#P.consumption_plot(shift=consumption, price=price, nonshift=0, shiftnames=app_names)
			plt.show()

		else:
			#print(res)
			print(res.message)
			print("Status: ", res.status)
			print("Minimized cost: %.3f" % res.fun)
		"""
