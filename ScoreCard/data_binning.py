# -*- coding:utf-8 -*-
__author__ = 'chunhui.zhang'
import sys
sys.path.append(r'E:\Code\Python_ML_Code\ScoreCard')
import load_data as ld
from config_params import *
import pandas as pd

rawdata_path = 'E:/ScoreCard/rawdata/'
dataset =ld.load_from_csv(rawdata_path,['pos_model_var_tbl_train',])
dataset_train = dataset[0]

if 1>0:
    raise MyException("the colums num of dataset_train and varibale_type is not equal")

print variable_type.loc['pos_cur_banlance','v_type'] == 'object'

if len(dataset_train.columns) == b.shape[0]:
    for vname in dataset_train.columns:
        dataset_train[vname] = dataset_train[vname].astype(b.loc[vname,'v_type'])
else:
    raise MyException("the colums num of dataset_train and varibale_type is not equal")


import sys
sys.path.append(r'E:\GitCode\Python_ML_Code\ScoreCard')
from config_params import *
import load_data as ld

dataset = ld.load_from_csv(raw_data_path,[train_file_name,])
dataset_train = dataset[0]
