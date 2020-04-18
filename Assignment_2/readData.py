"""
IN5410 - Energy informatics | Assignment 2

This file...
"""
import os, random, xlsxwriter, sys, argparse

import numpy               	as np
import pandas               as pd
import seaborn              as sns

# -----------------------------------------------------------------------------

def Data(filename='/TrainData.csv'):
    """
    Function for reading csv files
    Input: Filename as a string
    https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.read_csv.html
    """
    cwd      = os.getcwd()
    fn       = cwd + filename
    nanDict  = {}
    Data     = pd.read_csv(fn, header=0, skiprows=0, index_col=0, na_values=nanDict)
    return Data
