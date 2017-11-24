# -*- coding:utf-8 -*-
__author__ = 'maomaochong'
import pandas as pd
import woe.config as config
import woe.feature_process as fp
import woe.eval as eval
import woe.GridSearch as gs
import numpy as np
import pickle
import time

"""
训练WOE规则
"""
def process_train_woe(infile_path=None,outfile_path=None,rst_path=None):
    print 'run into process_train_woe: \n',time.asctime(time.localtime(time.time()))
    config_path = r'E:\Code\Python_ML_Code\cs_model\config\config_cs_model.csv'
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

feature_detail01,rst01 = process_train_woe(infile_path=r'E:\ScoreCard\cs_model\raw_data\m1_rsx_cs_unify_model_features_201701.csv'
                  ,outfile_path=r'E:\ScoreCard\cs_model\eval\cs_m1_pos_201701_features_detail.csv'
                  ,rst_path=r'E:\ScoreCard\cs_model\gendata\cs_m1_pos_woe_rule_201701.pkl')

feature_detail02,rst02 = process_train_woe(infile_path=r'E:\ScoreCard\cs_model\raw_data\m1_rsx_cs_unify_model_features_201702.csv'
                  ,outfile_path=r'E:\ScoreCard\cs_model\eval\cs_m1_pos_201702_features_detail.csv'
                  ,rst_path=r'E:\ScoreCard\cs_model\gendata\cs_m1_pos_woe_rule_201702.pkl')

feature_detail03,rst03 = process_train_woe(infile_path=r'E:\ScoreCard\cs_model\raw_data\m1_rsx_cs_unify_model_features_201703.csv'
                  ,outfile_path=r'E:\ScoreCard\cs_model\eval\cs_m1_pos_201703_features_detail.csv'
                  ,rst_path=r'E:\ScoreCard\cs_model\gendata\cs_m1_pos_woe_rule_201703.pkl')

feature_detail04,rst04 = process_train_woe(infile_path=r'E:\ScoreCard\cs_model\raw_data\m1_rsx_cs_unify_model_features_201704.csv'
                  ,outfile_path=r'E:\ScoreCard\cs_model\eval\cs_m1_pos_201704_features_detail.csv'
                  ,rst_path=r'E:\ScoreCard\cs_model\gendata\cs_m1_pos_woe_rule_201704.pkl')

feature_detail05,rst05 = process_train_woe(infile_path=r'E:\ScoreCard\cs_model\raw_data\m1_rsx_cs_unify_model_features_201705.csv'
                  ,outfile_path=r'E:\ScoreCard\cs_model\eval\cs_m1_pos_201705_features_detail.csv'
                  ,rst_path=r'E:\ScoreCard\cs_model\gendata\cs_m1_pos_woe_rule_201705.pkl')

feature_detail06,rst06 = process_train_woe(infile_path=r'E:\ScoreCard\cs_model\raw_data\m1_rsx_cs_unify_model_features_201706.csv'
                  ,outfile_path=r'E:\ScoreCard\cs_model\eval\cs_m1_pos_201706_features_detail.csv'
                  ,rst_path=r'E:\ScoreCard\cs_model\gendata\cs_m1_pos_woe_rule_201706.pkl')

feature_detail07,rst07 = process_train_woe(infile_path=r'E:\ScoreCard\cs_model\raw_data\m1_rsx_cs_unify_model_features_201707.csv'
                  ,outfile_path=r'E:\ScoreCard\cs_model\eval\cs_m1_pos_201707_features_detail.csv'
                  ,rst_path=r'E:\ScoreCard\cs_model\gendata\cs_m1_pos_woe_rule_201707.pkl')

feature_detail08,rst08 = process_train_woe(infile_path=r'E:\ScoreCard\cs_model\raw_data\m1_rsx_cs_unify_model_features_201708.csv'
                  ,outfile_path=r'E:\ScoreCard\cs_model\eval\cs_m1_pos_201708_features_detail.csv'
                  ,rst_path=r'E:\ScoreCard\cs_model\gendata\cs_m1_pos_woe_rule_201708.pkl')

feature_detail09,rst09 = process_train_woe(infile_path=r'E:\ScoreCard\cs_model\raw_data\m1_rsx_cs_unify_model_features_201709.csv'
                  ,outfile_path=r'E:\ScoreCard\cs_model\eval\cs_m1_pos_201709_features_detail.csv'
                  ,rst_path=r'E:\ScoreCard\cs_model\gendata\cs_m1_pos_woe_rule_201709.pkl')

"""
进行WOE转换
"""
def process_woe_trans(in_data_path=None,rst_path=None,out_path=None):
    config_path = r'E:\Code\Python_ML_Code\cs_model\config\config_cs_model.csv'
    data_path = in_data_path
    cfg = config.config()
    cfg.load_file(config_path, data_path)

    for var in [tmp for tmp in cfg.bin_var_list if tmp in list(cfg.dataset_train.columns)]:
        # fill null
        cfg.dataset_train.loc[cfg.dataset_train[var].isnull(), (var)] = -1

    for var in [tmp for tmp in cfg.discrete_var_list if tmp in list(cfg.dataset_train.columns)]:
        # fill null
        cfg.dataset_train.loc[cfg.dataset_train[var].isnull(), (var)] = 'missing'

    fp.change_feature_dtype(cfg.dataset_train, cfg.variable_type)

    output = open(rst_path, 'rb')
    rst = pickle.load(output)
    output.close()

    # Training dataset Woe Transformation
    for r in rst:
        cfg.dataset_train[r.var_name] = fp.woe_trans(cfg.dataset_train[r.var_name], r)

    cfg.dataset_train.to_csv(out_path)


in_data_path_list = [
    r'E:\ScoreCard\cs_model\raw_data\m1_rsx_cs_unify_model_features_201701.csv',
    r'E:\ScoreCard\cs_model\raw_data\m1_rsx_cs_unify_model_features_201702.csv',
    r'E:\ScoreCard\cs_model\raw_data\m1_rsx_cs_unify_model_features_201703.csv',
    r'E:\ScoreCard\cs_model\raw_data\m1_rsx_cs_unify_model_features_201704.csv',
    r'E:\ScoreCard\cs_model\raw_data\m1_rsx_cs_unify_model_features_201705.csv',
    r'E:\ScoreCard\cs_model\raw_data\m1_rsx_cs_unify_model_features_201706.csv',
    r'E:\ScoreCard\cs_model\raw_data\m1_rsx_cs_unify_model_features_201707.csv',
    r'E:\ScoreCard\cs_model\raw_data\m1_rsx_cs_unify_model_features_201708.csv',
    r'E:\ScoreCard\cs_model\raw_data\m1_rsx_cs_unify_model_features_201709.csv'
]

rst_path_list = [
    r'E:\ScoreCard\cs_model\gendata\cs_m1_pos_woe_rule_201701.pkl',
    r'E:\ScoreCard\cs_model\gendata\cs_m1_pos_woe_rule_201702.pkl',
    r'E:\ScoreCard\cs_model\gendata\cs_m1_pos_woe_rule_201703.pkl',
    r'E:\ScoreCard\cs_model\gendata\cs_m1_pos_woe_rule_201704.pkl',
    r'E:\ScoreCard\cs_model\gendata\cs_m1_pos_woe_rule_201705.pkl',
    r'E:\ScoreCard\cs_model\gendata\cs_m1_pos_woe_rule_201706.pkl',
    r'E:\ScoreCard\cs_model\gendata\cs_m1_pos_woe_rule_201707.pkl',
    r'E:\ScoreCard\cs_model\gendata\cs_m1_pos_woe_rule_201708.pkl',
    r'E:\ScoreCard\cs_model\gendata\cs_m1_pos_woe_rule_201709.pkl'
]

# range(rst_path_list.__len__())
for i in [7,8]:
    for j in range(in_data_path_list.__len__()):
        print '[START]',time.asctime(time.localtime(time.time()))
        print 'Processing '+'cs_m1_pos_woe_transed_rule_20170'+str(i+1)+'_features_20170'+str(j+1)+':'
        woed_out_path =  'E:\\ScoreCard\\cs_model\\gendata\\' + 'cs_m1_pos_woe_transed_rule_20170'+str(i+1)\
                         +'_features_20170'+str(j+1)+'.csv'
        process_woe_trans(in_data_path=in_data_path_list[j]
                          ,rst_path=rst_path_list[i]
                          ,out_path=woed_out_path)
        print '[END]', time.asctime(time.localtime(time.time()))

"""
Train Logistic Regression Model
"""
dataset_path_list = [
    'cs_m1_pos_woe_transed_rule_201701_features_201701.csv',
    'cs_m1_pos_woe_transed_rule_201702_features_201702.csv',
    'cs_m1_pos_woe_transed_rule_201703_features_201703.csv',
    'cs_m1_pos_woe_transed_rule_201704_features_201704.csv',
    'cs_m1_pos_woe_transed_rule_201705_features_201705.csv',
    'cs_m1_pos_woe_transed_rule_201706_features_201706.csv',
    'cs_m1_pos_woe_transed_rule_201707_features_201707.csv',
    'cs_m1_pos_woe_transed_rule_201708_features_201708.csv',
    'cs_m1_pos_woe_transed_rule_201709_features_201709.csv'
]

df_coef_path_list = [
    'cs_m1_pos_coef_path_rule_201701_features_201701.csv',
    'cs_m1_pos_coef_path_rule_201702_features_201702.csv',
    'cs_m1_pos_coef_path_rule_201703_features_201703.csv',
    'cs_m1_pos_coef_path_rule_201704_features_201704.csv',
    'cs_m1_pos_coef_path_rule_201705_features_201705.csv',
    'cs_m1_pos_coef_path_rule_201706_features_201706.csv',
    'cs_m1_pos_coef_path_rule_201707_features_201707.csv',
    'cs_m1_pos_coef_path_rule_201708_features_201708.csv',
    'cs_m1_pos_coef_path_rule_201709_features_201709.csv'
]

c_list = []
ks_list = []

for i in range(dataset_path_list.__len__()):
    print '[START]',time.asctime(time.localtime(time.time()))
    dataset_path = 'E:\\ScoreCard\\cs_model\\gendata\\' + dataset_path_list[i]
    df_coef_path = 'E:\\ScoreCard\\cs_model\\eval\\' + df_coef_path_list[i]
    pic_coefpath_title = 'cs_m1_pos_coef_path_rule_20170'+str(i+1)+'_features_20170'+str(i+1)
    pic_coefpath = 'E:\\ScoreCard\\cs_model\\eval\\' + 'cs_m1_pos_coef_path_rule_20170'+str(i+1)\
                   +'_features_20170'+str(i+1)+'.png'
    pic_performance_title = 'cs_m1_pos_performance_path_rule_20170'+str(i+1)+'_features_20170'+str(i+1)
    pic_performance = 'E:\\ScoreCard\\cs_model\\eval\\' + 'cs_m1_pos_performance_path_rule_20170'+str(i+1)\
                      +'_features_20170'+str(i+1)+'.png'

    dataset_train = pd.read_csv(dataset_path)
    cfg = pd.read_csv(r'E:\Code\Python_ML_Code\cs_model\config\config_cs_model.csv')
    candidate_var_list = cfg[cfg['is_modelfeature'] == 1]['var_name']

    b = [var for var in dataset_train.columns if sum(dataset_train[var].isnull()) == 0]
    candidate_var_list = list(set(candidate_var_list).intersection(set(b)))

    X_train = dataset_train[candidate_var_list]
    y_train = dataset_train['target']

    c,ks = gs.grid_search_lr_c(X_train,y_train,df_coef_path,pic_coefpath_title,pic_coefpath
                               ,pic_performance_title,pic_performance)

    c_list.append(c)
    ks_list.append(ks)

