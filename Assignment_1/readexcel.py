import os, random, xlsxwriter
import numpy               as np
import pandas              as pd


#------------------------------------------------------------------------------
cwd      = os.getcwd()
filename = cwd + '/energy_use.xlsx'
nanDict  = {}
df       = pd.read_excel(filename, header=0, skiprows=0, index_col=0, na_values=nanDict)

print(df)


header_names = list(df)
print(header_names)

print(df['Alpha']['EV'])