# -*- coding:utf-8 -*-
__author__ = 'maomaochong'
import pandas as pd
import woe.config as config

# get columns names list
config_path = r'E:\Code\Python_ML_Code\cs_model\config\config_cs_daily_model.csv'
cfg = config.config()
cfg.load_file(config_path)
col_dataset = pd.read_csv(config_path)
col_name_list = col_dataset.var_name

for feature_name in col_name_list[1:]:
    feature_path = 'E:\\ScoreCard\\cs_model\\cs_m1_pos_model_daily\\raw_data\\dataset_split_by_cols\\'
    feature_path = feature_path + feature_name + '.csv'
    try:
        feature = pd.read_csv(feature_path)
        print('%s\t :\t%d' %(feature_name,feature.shape[0]))
    except:
        print 'error:\t',feature_name