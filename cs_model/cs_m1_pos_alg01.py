# -*- coding:utf-8 -*-
__author__ = 'maomaochong'
import pandas as pd
import woe.config as config
import woe.feature_process as fp
import woe.eval as eval
import numpy as np
import pickle
import time

def process_train_woe(infile_path=None,outfile_path=None,rst_path=None):
    print 'run into process_train_woe: \n',time.asctime(time.localtime(time.time()))
    config_path = r'E:\Code\Python_ML_Code\cs_model\config\config_cs_model.csv'
    data_path = infile_path
    cfg = config.config()
    cfg.load_file(config_path,data_path)

    for var in [tmp for tmp in cfg.bin_var_list if tmp in list(cfg.dataset_train.columns)]:
        # fill null
        cfg.dataset_train.loc[cfg.dataset_train[var].isnull(), (var)] = -1

    # change feature dtypes
    fp.change_feature_dtype(cfg.dataset_train, cfg.variable_type)
    rst = []

    # process woe transformation of continuous variables
    print 'process woe transformation of continuous variables: \n',time.asctime(time.localtime(time.time()))
    print 'cfg.global_bt',cfg.global_bt
    print 'cfg.global_gt', cfg.global_gt

    for var in [tmp for tmp in cfg.bin_var_list if tmp in list(cfg.dataset_train.columns)]:
        rst.append(fp.proc_woe_continuous(cfg.dataset_train,var,cfg.global_bt,cfg.global_gt,cfg.min_sample,alpha=0.05))

    # process woe transformation of discrete variables
    print 'process woe transformation of discrete variables: \n',time.asctime(time.localtime(time.time()))
    for var in [tmp for tmp in cfg.discrete_var_list if tmp in list(cfg.dataset_train.columns)]:
        # fill null
        cfg.dataset_train.loc[cfg.dataset_train[var].isnull(), (var)] = 'missing'
        rst.append(fp.proc_woe_discrete(cfg.dataset_train,var,cfg.global_bt,cfg.global_gt,cfg.min_sample,alpha=0.05))

    eval.eval_feature_detail(rst, outfile_path)

    print 'save woe transformation rule into pickle: \n',time.asctime(time.localtime(time.time()))
    output = open(rst_path, 'wb')
    pickle.dump(rst,output)
    output.close()

process_train_woe(infile_path=r'E:\ScoreCard\cs_model\m1_rsx_cs_unify_model_features_201701.csv'
                  ,outfile_path=r'E:\ScoreCard\cs_model\eval\cs_m1_pos_201701_features_detail.csv'
                  ,rst_path='E:\\Code\\ScoreCard\gendata\cs_m1_pos_woe_rule_201701.pkl')

process_train_woe(infile_path=r'E:\ScoreCard\cs_model\m1_rsx_cs_unify_model_features_201702.csv'
                  ,outfile_path=r'E:\ScoreCard\cs_model\eval\cs_m1_pos_201702_features_detail.csv'
                  ,rst_path=r'E:\ScoreCard\cs_model\gendata\cs_m1_pos_woe_rule_201702.pkl')

process_train_woe(infile_path=r'E:\ScoreCard\cs_model\m1_rsx_cs_unify_model_features_201703.csv'
                  ,outfile_path=r'E:\ScoreCard\cs_model\eval\cs_m1_pos_201703_features_detail.csv'
                  ,rst_path=r'E:\ScoreCard\cs_model\gendata\cs_m1_pos_woe_rule_201703.pkl')

process_train_woe(infile_path=r'E:\ScoreCard\cs_model\m1_rsx_cs_unify_model_features_201704.csv'
                  ,outfile_path=r'E:\ScoreCard\cs_model\eval\cs_m1_pos_201704_features_detail.csv'
                  ,rst_path='E:\\Code\\ScoreCard\gendata\cs_m1_pos_woe_rule_201704.pkl')

process_train_woe(infile_path=r'E:\ScoreCard\cs_model\m1_rsx_cs_unify_model_features_201705.csv'
                  ,outfile_path=r'E:\ScoreCard\cs_model\eval\cs_m1_pos_201705_features_detail.csv'
                  ,rst_path='E:\\Code\\ScoreCard\gendata\cs_m1_pos_woe_rule_201705.pkl')

process_train_woe(infile_path=r'E:\ScoreCard\cs_model\m1_rsx_cs_unify_model_features_201706.csv'
                  ,outfile_path=r'E:\ScoreCard\cs_model\eval\cs_m1_pos_201706_features_detail.csv'
                  ,rst_path='E:\\Code\\ScoreCard\gendata\cs_m1_pos_woe_rule_201706.pkl')

process_train_woe(infile_path=r'E:\ScoreCard\cs_model\m1_rsx_cs_unify_model_features_201707.csv'
                  ,outfile_path=r'E:\ScoreCard\cs_model\eval\cs_m1_pos_201707_features_detail.csv'
                  ,rst_path='E:\\Code\\ScoreCard\gendata\cs_m1_pos_woe_rule_201707.pkl')

process_train_woe(infile_path=r'E:\ScoreCard\cs_model\m1_rsx_cs_unify_model_features_201708.csv'
                  ,outfile_path=r'E:\ScoreCard\cs_model\eval\cs_m1_pos_201708_features_detail.csv'
                  ,rst_path='E:\\Code\\ScoreCard\gendata\cs_m1_pos_woe_rule_201708.pkl')

process_train_woe(infile_path=r'E:\ScoreCard\cs_model\m1_rsx_cs_unify_model_features_201709.csv'
                  ,outfile_path=r'E:\ScoreCard\cs_model\eval\cs_m1_pos_201709_features_detail.csv'
                  ,rst_path='E:\\Code\\ScoreCard\gendata\cs_m1_pos_woe_rule_201709.pkl')
