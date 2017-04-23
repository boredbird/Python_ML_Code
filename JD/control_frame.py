# -*- coding:utf-8 -*-
__author__ = 'maomaochong'
import sys
sys.path.append('E:/Code/Python_ML_Code/JD')
import pandas as pd
import load_data

raw_data_path = 'E:/Code/Python_ML_Code/JD/raw_data/'
file = ['JData_Comment','JData_Product','JData_User']

#load row data file from csv
row_data = load_data.load_from_csv(raw_data_path,file)

print   1
# dframe = pd.DataFrame({'total_bill': [16.99, 10.34, 23.68, 23.68, 24.59],
#                    'tip': [1.01, 1.66, 3.50, 3.31, 3.61],
#                    'sex': ['Female', 'Male', 'Male', 'Male', 'Female']})

#load dataframe into csv:
# load_data.load_into_csv(raw_data_path,df,'aa')

#load dataframe into mysql:
# load_data.load_into_mysql(dframe,'aa')

#load dataframe from mysql:
# tableframe = load_data.load_from_mysql('aa')

