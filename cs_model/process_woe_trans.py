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
import os

"""
进行WOE转换
"""
def process_woe_trans(cfg=None,in_data_path=None,rst_path=None,out_path=None):
    # config_path = r'E:\Code\Python_ML_Code\cs_model\config\config_cs_daily_model.csv'
    # data_path = in_data_path
    # cfg = config.config()
    # cfg.load_file(config_path, data_path)

    dataset = pd.read_csv(in_data_path)
    for var in [tmp for tmp in cfg.bin_var_list if tmp in list(dataset.columns)]:
        # fill null
        dataset.loc[dataset[var].isnull(), (var)] = -1

    for var in [tmp for tmp in cfg.discrete_var_list if tmp in list(dataset.columns)]:
        # fill null
        dataset.loc[dataset[var].isnull(), (var)] = 'missing'

    fp.change_feature_dtype(dataset, cfg.variable_type)

    output = open(rst_path, 'rb')
    rst = pickle.load(output)
    output.close()

    # Training dataset Woe Transformation
    r = rst[0]
    dataset[r.var_name] = fp.woe_trans(dataset[r.var_name], r)
    dataset.to_csv(out_path)
    print('%s\tSUCCESS EXPORT FILE: \n%s' %(time.asctime(time.localtime(time.time())),out_path))


def single_process_woe_trans(var):
    print('%s\tPROC WOE TRANS: \t%s' %(time.asctime(time.localtime(time.time())),var))
    # target_path = r'E:\ScoreCard\cs_model\cs_m1_pos_model_daily\raw_data\dataset_split_by_cols\target.csv'
    # target = pd.read_csv(target_path)
    #
    # cfg.global_bt = sum(target.target)
    # cfg.global_gt = target.shape[0] - cfg.global_bt
    # cfg.min_sample = int(target.shape[0]*0.05)

    in_data_path = 'E:\\ScoreCard\\cs_model\\cs_m1_pos_model_daily\\raw_data\\dataset_split_by_cols\\' + var + '.csv'
    rst_path = 'E:\\ScoreCard\\cs_model\\cs_m1_pos_model_daily\\gendata\\WOE_Rule\\' + var + '.pkl'
    out_path = 'E:\\ScoreCard\\cs_model\\cs_m1_pos_model_daily\\gendata\\' + var + '_woe_transed.csv'
    return process_woe_trans(cfg,in_data_path,rst_path,out_path)
    # return process_woe_trans(in_data_path,rst_path,out_path)

if __name__ == "__main__":
    # get columns names list
    config_path = r'E:\Code\Python_ML_Code\cs_model\config\config_cs_daily_model.csv'
    cfg = config.config()
    cfg.load_file(config_path)
    col_dataset = pd.read_csv(config_path)
    col_name_list = col_dataset.var_name

    feature_list = list(cfg.bin_var_list)
    feature_list.extend(list(cfg.discrete_var_list))

    for i in range(feature_list.__len__()):
        try:
            result = single_process_woe_trans(feature_list[i])
            # result = single_process(feature_list[i])
        except Exception:
            print '[error]',feature_list[i]
