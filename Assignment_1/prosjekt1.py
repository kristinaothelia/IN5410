import os, random, xlsxwriter, sys, argparse

import matplotlib.pyplot 	as plt
import numpy               	as np
import pandas               as pd
import functions 		   	as func
import plotting 			as P

from scipy.optimize 		import linprog
from random 				import seed

# Python 3.7.4
#------------------------------------------------------------------------------
df 	  = func.Get_df(file_name='/energy_use.xlsx')	# Get data for appliances
hours = 24

nr_non_shiftable = len(df[df['Shiftable'] == 0])

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

			plt.show()

		else:
			#print(res)
			print(res.message)
			print("Status: ", res.status)
			print("Minimized cost: %.3f" % res.fun)


	elif Task3 == True:

		print("--"*40); print("Task 3"); print("--"*40)
		# ---------------------------------------------------------------------
		households    = 30

		# Fill up arrays with total consumption for all households
		Total_con_n = np.zeros(hours)		# Non-shiftable
		Total_con_s = np.zeros(hours)		# Shiftable

		# Get pricing scheme. ToU (Time-of-Use) or RTP (Real-Time-Pricing)
		price = func.Get_price(hours, seed=seed, ToU=False)
		cost  = 0

		# Skal alle husholdningene ha samme pris, eller skal dette genereres ulikt?
		EV_number = 0
		for i in range(households):
			df 	  = func.Get_df(file_name='/energy_use.xlsx')	# Get data for appliances

			n_app, alpha, beta, length, non_shiftable, non_shiftable_names, \
			shiftable_combined, shiftable_c_names, EV_nr \
			= func.applications_Task3(df, households)

			if EV_nr == 1:
				EV_number += 1

			# Creating intervals
			intervals = func.interval_severeal_app(n_app, length, alpha, beta, shuffle=False)

			df = pd.concat((non_shiftable, shiftable_combined))

			# Make vriables for linprog. c, A_eq, b_eq, A_ub, b_ub
			c, A_eq, b_eq, A_ub, b_ub = func.linprog_input(df, n_app, price, intervals, hours)

			# Make linprog calculations
			res 		= linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=(0,None))
			consumption = res.x.reshape(n_app, hours)

			non_s_con = consumption[:nr_non_shiftable]
			shift_con = consumption[nr_non_shiftable:]

			non_shift_tot = np.sum(non_s_con, axis=0)
			#print('Total hourly consumption for non-shiftable app.', '\n', non_shift_tot)

			shift_tot    = np.sum(shift_con, axis=0)
			#print('Total hourly consumption for shiftable app.', '\n', shift_tot)

			Total_con_n  += non_shift_tot
			Total_con_s  += shift_tot

			# Lagre bilde for hver husholdning??
			#plt.savefig("Household%g" %(i+1))

			cost += res.fun

			#print(res.message)
			#print("Status: ", res.status)
			print("House %g, Minimized cost: %.3f" % (i+1, res.fun))


		if Plot == True:

			P.Make_p_hist(df, price)
			P.consumption_plot_Task3(price=price, EV=EV_number, \
									 app=Total_con_s, 			\
									 app_names='Shiftable applications', 	\
									 non_app=Total_con_n, 		\
									 non_app_names='Non-shiftable applications')
			plt.show()

		else:

			print('Neighborhood:')
			print(Total_con_n, '----')
			print(Total_con_s, '--')
			print('Cost: ', cost)
			print('EVs:  ', EV_number)
