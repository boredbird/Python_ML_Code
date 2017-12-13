# -*- coding:utf-8 -*-
__author__ = 'maomaochong'
import pandas as pd
import woe.config as config
import woe.feature_process as fp
import woe.eval as eval
import pickle
import time
import csv
import numpy as np
from multiprocessing import Pool

# get columns names list
config_path = r'E:\Code\Python_ML_Code\cs_model\config\config_cs_daily_model.csv'
cfg = config.config()
cfg.load_file(config_path)
col_dataset = pd.read_csv(config_path)
col_name_list = col_dataset.var_name

"""
拆分数据集 by cols
"""
# def a function for put incremental data block into csv
def block_to_csv(block,col_name_list):
    # for i in range(block[0].__len__()):
    for i in range(col_name_list.__len__()):
        col_name = col_name_list[i]
        tmp = []
        tmp.extend([line[i].strip('"') for line in block])

        split_path = 'E:\\ScoreCard\\cs_model\\cs_m1_pos_model_daily\\raw_data\\dataset_split_by_cols\\'
        split_path_tmp = split_path + col_name + '.csv'

        with open(split_path_tmp,'ab') as f:
            # writer=csv.writer(f)
            for line in range(0,tmp.__len__()):# not included row name
                f.write('%s\n' % tmp[line])

        f.close()

# main function to split dataset by cols
infile_path=r'E:\ScoreCard\cs_model\cs_m1_pos_model_daily\raw_data\m1_rsx_cs_unify_model_features_201705_daily.csv'
count = 0
block = []
with open(infile_path, 'rb') as f:
    for line in f:
        if line.strip('"').strip().split(",").__len__() == 130: # check !!!!!
            block.append(line.strip('"').strip().split(","))
            count += 1
            if count %100000==0 or count >= 11932695: # total row nums:11932695
                # call function to write block into csv
                print time.asctime(time.localtime(time.time())),count
                block_to_csv(block,col_name_list)
                block = [] # clear the tmp block after insert

