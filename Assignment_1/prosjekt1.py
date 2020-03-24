import os, random, xlsxwriter, sys, argparse, warnings, excel2img

import matplotlib.pyplot 	as plt
import numpy               	as np
import pandas               as pd
import functions 		   	as func
import plotting 			as P
import seaborn as sns

from scipy.optimize 		import linprog
from random 				import seed
from pandas.plotting 		import table 


#from StyleFrame import StyleFrame, Styler, utils
# Python 3.7.4
#------------------------------------------------------------------------------
df 	  = func.Get_df(file_name='/energy_use.xlsx')	# Get data for appliances
hours = 24

print(df)

nr_non_shiftable = len(df[df['Shiftable'] == 0])

if __name__ == '__main__':

	parser = argparse.ArgumentParser(description="Minimize energy cost")

	group = parser.add_mutually_exclusive_group()
	group.add_argument('-1', '--Task1', action="store_true", help="3 shiftable appliances")
	group.add_argument('-2', '--Task2', action="store_true", help="Shiftable and non-shiftable appliances")
	group.add_argument('-3', '--Task3', action="store_true", help="30 households")

	# Optional argument for plotting
	parser.add_argument('-X', '--plot', action='store_true', help="Plotting", required=False)

	# Optional argument for printing out possible warnings
	parser.add_argument('-W', '--warnings', action='store_true', help="Warnings", required=False)

	if len(sys.argv) <= 1:
		sys.argv.append('--help')

	args  = parser.parse_args()

	Task1    = args.Task1
	Task2    = args.Task2
	Task3    = args.Task3
	Plot     = args.plot
	Warnings = args.warnings

	if not Warnings:
		# If the argument -W / --warnings is provided, 
		# any warnings will be printed in the terminal
		warnings.filterwarnings("ignore")

	if Task1 == True:

		print(''); print("=="*44)
		print("Task 1: A simple household with 3 shiftable appliances (washing machine, EV, dishwasher)")
		print("=="*44, '\n')

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
			print("Minimized cost: %.3f NOK" % res.fun)


	elif Task2 == True:

		print(''); print("=="*39)
		print("Task 2: A household with shiftable and non_shiftable appliances and RTP scheme")
		print("=="*39, '\n')

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
			print("Minimized cost: %.3f NOK" % res.fun)


	elif Task3 == True:

		print(''), print("="*79)
		print("Task 3: A small neighborhood where only a fraction of the households owns an EV")
		print("="*79, '\n')

		households  = 30

		# Fill up arrays with total consumption for all households
		Total_con_n = np.zeros(hours)		# Non-shiftable
		Total_con_s = np.zeros(hours)		# Shiftable

		# Get pricing scheme. ToU (Time-of-Use) or RTP (Real-Time-Pricing)
		price = func.Get_price(hours, seed=seed, ToU=False)
		cost  = 0

		# Skal alle husholdningene ha samme pris, eller skal dette genereres ulikt?
		EV_number	 = 0
		house_nr     = []
		cost_nr  	 = []
		hav_nonshift = []
		hav_shift    = []
		EV_yes_no    = []

		for i in range(households):
			df 	  = func.Get_df(file_name='/energy_use.xlsx')	# Get data for appliances

			n_app, alpha, beta, length, non_shiftable, non_shiftable_names, \
			shiftable_combined, shiftable_c_names, EV_nr \
			= func.applications_Task3(df, households)

			if EV_nr == 1:
				EV_number += 1
				EV_yes_no.append('Yes')
			else:
				EV_yes_no.append('No')

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

			# Average hourly consumption of the household, ha med dette??
			hav_nonshift.append(np.sum(non_shift_tot)/hours)
			hav_shift.append(np.sum(non_shift_tot)/hours)

			Total_con_n  += non_shift_tot
			Total_con_s  += shift_tot

			# Lagre bilde for hver husholdning??
			#plt.savefig("Household%g" %(i+1))

			cost += res.fun

			#print(res.message)
			#print("Status: ", res.status)
			if i < 9:
				print("House %g,  Minimized cost: %.3f NOK" % (i+1, res.fun))
			else:
				print("House %g, Minimized cost: %.3f NOK" % (i+1, res.fun))

			#house_nr.append('House ' + '%g' %(i+1))
			house_nr.append(i+1)
			cost_nr.append('%.3f' %res.fun)


		# skal vi ha noe saant? Kanskje med andre ting?  EV, shiftable/not
		# legge i funksjon
		list_of_tuples = list(zip(hav_nonshift, hav_shift, cost_nr, EV_yes_no))
		result_table   = pd.DataFrame(list_of_tuples,index=house_nr,\
					     columns = ['Non-shiftable [kW/h]', 'Shiftable [kW/h]', 'Minimized cost [NOK]', 'EV'])


		writer   = pd.ExcelWriter('result_table.xlsx', engine='xlsxwriter')
		result_table.to_excel(writer, sheet_name='task3', float_format="%.3f", index_label='House')
		workbook = writer.book


		#full_border = workbook.add_format({'border':1, 'border_color':'#000000'})
		# cant get border lines at the edges..... 
		format1     = workbook.add_format({'align': 'center','bold': False})
		format2     = workbook.add_format({'border':1})
		worksheet   = writer.sheets['task3']
		worksheet.set_column('B:E', None, format1)
		worksheet.set_column('B:D', 18, None)
		#worksheet.write('A3', '', format2)
		border_fmt = workbook.add_format({'bottom':2, 'top':2, 'left':5, 'right':1})
		#worksheet.conditional_format(xlsxwriter.utility.xl_range(0, 0, len(result_table), len(result_table.columns)), {'type': 'no_errors', 'format': border_fmt})
		worksheet.conditional_format(xlsxwriter.utility.xl_range(0, 0, 4, 0), {'type': 'no_errors', 'format': border_fmt})
		writer.save()


		#result_table.to_excel('result_table.xlsx', float_format="%.3f", index_label='House', engine='xlsxwriter')

		# Creates a table which can be imported to a latex document
		latex_table = result_table.to_latex('latex_table.tex', float_format="%.3f")

		# Creates a png image of the result table
		excel2img.export_img('result_table.xlsx','somesome.png')  # pip install excel2img


		if Plot == True:

			P.Make_p_hist(df, price)
			P.consumption_plot_Task3(price=price, EV=EV_number, \
									 app=Total_con_s, 			\
									 app_names='Shiftable applications', 	\
									 non_app=Total_con_n, 		\
									 non_app_names='Non-shiftable applications')
			plt.show()

		else:

			print(''); print('-'*47)
			print('Consumption and total cost for the Neighborhood')
			print('-'*47, '\n')
			print('Consumption of the non-shiftable appliances for each household:', '\n')
			print(Total_con_n, '\n')
			print('Consumption of the shiftable appliances for each household:', '\n')
			print(Total_con_s, '\n')
			print('Total cost for the neighborhood: %.3f NOK' %cost)
			print('Number of EVs: ', EV_number)


'''
# mer fancy excel fil
writer = pd.ExcelWriter('result_table.xlsx', engine='xlsxwriter')
		result_table.to_excel(writer, sheet_name='task3', float_format="%.3f", index_label='House', startrow=1, header=False)
		workbook = writer.book
		worksheet = writer.sheets['task3']

		# Add a header format.
		header_format = workbook.add_format({'bold': True,'text_wrap': True,'valign': 'center','fg_color': '#D7E4BC','border': 1})
		# Write the column headers with the defined format.
		for col_num, value in enumerate(result_table.columns.values):
			worksheet.write(0, col_num + 1, value, header_format)
			
		writer.save()
'''


'''
		writer = pd.ExcelWriter('result_table.xlsx', engine='xlsxwriter')
		result_table.to_excel(writer, sheet_name='task3', float_format="%.3f", index_label='House')
		workbook = writer.book
		full_border = workbook.add_format({'border':1, 'border_color':'#000000'})
		worksheet = writer.sheets['task3']
		worksheet.write("B", None, full_border)
		worksheet.set_column('B:D', 18)
		writer.save()
		'''

		#result_table.style.set_table_styles([{'selector':'','props':[('border','4px solid #7a7')]}])

'''
writer = StyleFrame.ExcelWriter("result_table.xlsx")
		result_table     = StyleFrame(result_table)

		result_table.apply_column_style(cols_to_style=result_table.columns, styler_obj=Styler(bg_color=utils.colors.white, bold=True, font=utils.fonts.arial,font_size=8),style_header=True)

		result_table.apply_headers_style(styler_obj=Styler(bg_color=utils.colors.blue, bold=True, font_size=8, font_color=utils.colors.white,number_format=utils.number_formats.general, protection=False))

		result_table.to_excel(writer, sheet_name='Sheet1')
		writer.save()
'''