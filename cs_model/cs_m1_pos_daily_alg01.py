# -*- coding:utf-8 -*-
__author__ = 'maomaochong'
import pandas as pd
import woe.config as config
import woe.feature_process as fp
import woe.eval as eval
import pickle
import time
import csv

config_path = r'E:\Code\Python_ML_Code\cs_model\config\config_cs_daily_model.csv'
cfg = config.config()
cfg.load_file(config_path)

infile_path=r'E:\ScoreCard\cs_model\cs_m1_pos_model_daily\raw_data\m1_rsx_cs_unify_model_features_201705_daily.csv'
dataset = pd.read_csv(infile_path)

for var in cfg.candidate_var_list:
    print time.asctime(time.localtime(time.time())),var
    outpath = 'E:\\ScoreCard\\cs_model\\cs_m1_pos_model_daily\\raw_data\\dataset_split_by_cols\\'
    dataset[[var,'target']].to_csv(outpath+var+'.csv')

"""
拆分数据集
"""
infile_path=r'E:\ScoreCard\cs_model\cs_m1_pos_model\raw_data\m1_rsx_cs_unify_model_features_201701.csv'
# dataset = pd.read_csv(infile_path)

count = 0

block = []
with open(infile_path, 'rb') as f:
    for line in f:
        count += 1
        block.append(line.strip('"').split(","))

col_name_list = [var for var in block[0]]
for i in range(block[0].__len__()):
    col_name = block[0][i+1]
    tmp = []
    tmp.append([line[i] for line in block])

    split_path = 'E:\\ScoreCard\\cs_model\\cs_m1_pos_model_daliy\\raw_data\\'
    split_path_tmp = split_path + col_name + '.csv'
    #python2可以用file替代open
    with open(split_path_tmp,"w") as csvfile:
        writer = csv.writer(csvfile)
        #先写入columns_name
        # writer.writerow(["index","a_name","b_name"])
        #写入多行用writerows
        writer.writerows(tmp)



"""
训练WOE规则
"""
def process_train_woe(infile_path=None,outfile_path=None,rst_path=None):
    print 'run into process_train_woe: \n',time.asctime(time.localtime(time.time()))
    config_path = r'E:\Code\Python_ML_Code\cs_model\config\config_cs_daily_model.csv'
    data_path = infile_path
    cfg = config.config()
    cfg.load_file(config_path,data_path)
    bin_var_list = [tmp for tmp in cfg.bin_var_list if tmp in list(cfg.dataset_train.columns)]

    for var in bin_var_list:
        # fill null
        cfg.dataset_train.loc[cfg.dataset_train[var].isnull(), (var)] = -1

    # change feature dtypes
    fp.change_feature_dtype(cfg.dataset_train, cfg.variable_type)
    rst = []

    # process woe transformation of continuous variables
    print 'process woe transformation of continuous variables: \n',time.asctime(time.localtime(time.time()))
    print 'cfg.global_bt',cfg.global_bt
    print 'cfg.global_gt', cfg.global_gt

    for var in bin_var_list:
        rst.append(fp.proc_woe_continuous(cfg.dataset_train,var,cfg.global_bt,cfg.global_gt,cfg.min_sample,alpha=0.05))

    # process woe transformation of discrete variables
    print 'process woe transformation of discrete variables: \n',time.asctime(time.localtime(time.time()))
    for var in [tmp for tmp in cfg.discrete_var_list if tmp in list(cfg.dataset_train.columns)]:
        # fill null
        cfg.dataset_train.loc[cfg.dataset_train[var].isnull(), (var)] = 'missing'
        rst.append(fp.proc_woe_discrete(cfg.dataset_train,var,cfg.global_bt,cfg.global_gt,cfg.min_sample,alpha=0.05))

    feature_detail = eval.eval_feature_detail(rst, outfile_path)

    print 'save woe transformation rule into pickle: \n',time.asctime(time.localtime(time.time()))
    output = open(rst_path, 'wb')
    pickle.dump(rst,output)
    output.close()

    return feature_detail,rst

# infile_path=r'E:\ScoreCard\cs_model\raw_data\m1_rsx_cs_unify_model_features_201701.csv'
# a = pd.read_csv(infile_path)
# b = a.loc[:10000,]
# b.to_csv(r'E:\ScoreCard\cs_model\raw_data\m1_rsx_cs_unify_model_features_201701_tmp.csv')

feature_detail,rst = process_train_woe(infile_path=r'E:\ScoreCard\cs_model\cs_m1_pos_model_daily\raw_data\m1_rsx_cs_unify_model_features_201705_daily.csv'
                  ,outfile_path=r'E:\ScoreCard\cs_model\eval\cs_m1_pos_201705_daily_features_detail.csv'
                  ,rst_path=r'E:\ScoreCard\cs_model\gendata\cs_m1_pos_daily_woe_rule_201705.pkl')
