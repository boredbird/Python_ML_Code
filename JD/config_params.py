# -*- coding:utf-8 -*-
__author__ = 'chunhui.zhang'

import  pandas as pd
import numpy as np

#raw data file path
raw_data_path = 'E:/Code/Python_ML_Code/JD/raw_data/'

#raw data file name
raw_file_name = [#'JData_Action_201602','JData_Action_201603','JData_Action_201604'
                'JData_Action_ALL',#'JData_Comment','JData_Product','JData_User'
                    ]

# split data path
split_data_path = 'E:/Code/Python_ML_Code/JD/split_data/'

# model path

# submission path
submission_path = 'E:/Code/Python_ML_Code/JD/submission/'

# global values
#>=  datetime  <
split_date = pd.DataFrame({'predictset': ['2016-03-17','2016-04-16','2016-04-16','2016-04-21'],
                   'testset': ['2016-03-12','2016-04-11','2016-04-11','2016-04-16'],
                    'trainset1': ['2016-03-02','2016-04-01','2016-04-01','2016-04-06'],
                    'trainset2': ['2016-02-26','2016-03-27','2016-03-27','2016-04-01'],
                   'trainset3': ['2016-02-16','2016-03-17','2016-03-17','2016-03-22'],
                    'trainset4': ['2016-02-01','2016-03-02','2016-03-02','2016-03-07']})
split_date = split_date.T
split_date.columns = ['feature_start_date','feature_end_date','lable_start_date','lable_end_date']
split_date.loc['trainset4',['feature_start_date']]

split_date['feature_start_date'] = split_date['feature_start_date'].astype(np.datetime64)
split_date['feature_end_date'] = split_date['feature_end_date'].astype(np.datetime64)
split_date['lable_start_date'] = split_date['lable_start_date'].astype(np.datetime64)
split_date['lable_end_date'] = split_date['lable_end_date'].astype(np.datetime64)

