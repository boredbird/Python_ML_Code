# -*- coding:utf-8 -*-
__author__ = 'maomaochong'
import pandas as pd
import woe.feature_process as fp
import woe.eval as eval
import woe.config as config
import pickle
import time

def fliter_dataset(infile_path,outfile_path):
    dataset = pd.read_csv(infile_path)
    dataset.drop('target',axis=1, inplace=True)
    dataset = dataset.rename(columns={'mark':'target'})
    dataset.to_csv(outfile_path,index=False)


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


if __name__ == '__main__':
    infile_path1 = r'E:\ScoreCard\cs_model\cs_m1_pos_model\raw_data\m1_rsx_cs_unify_model_features_20170910.csv'
    outfile_path1 = r'E:\ScoreCard\cs_model\cs_m1_pos_model\m1_rsx_cs_unify_model_features_20170910.csv'
    fliter_dataset(infile_path1,outfile_path1)

    infile_path2 = r'E:\ScoreCard\cs_model\cs_m1_pos_model\raw_data\m1_rsx_cs_unify_model_features_20171011.csv'
    outfile_path2 = r'E:\ScoreCard\cs_model\cs_m1_pos_model\m1_rsx_cs_unify_model_features_20171011.csv'
    fliter_dataset(infile_path2,outfile_path2)


    # 训练woe规则
    config_path = r'E:\Code\Python_ML_Code\cs_model\config\config_cs_model.csv'
    feature_detail_file_path1 = r'E:\ScoreCard\cs_model\cs_m1_pos_model\monitor\cs_m1_pos_20170910_monitor_features_detail.csv'
    woe_pkl_file_path1 = r'E:\ScoreCard\cs_model\cs_m1_pos_model\monitor\cs_m1_pos_woe_rule_monitor_20170910.pkl'
    feature_detail,rst = process_train_woe(infile_path=outfile_path1
                                       ,outfile_path=feature_detail_file_path1
                                       ,rst_path=woe_pkl_file_path1)

    feature_detail_file_path2 = r'E:\ScoreCard\cs_model\cs_m1_pos_model\monitor\cs_m1_pos_20171011_monitor_features_detail.csv'
    woe_pkl_file_path2 = r'E:\ScoreCard\cs_model\cs_m1_pos_model\monitor\cs_m1_pos_woe_rule_monitor_20171011.pkl'
    feature_detail,rst = process_train_woe(infile_path=outfile_path2
                                           ,outfile_path=feature_detail_file_path2
                                           ,rst_path=woe_pkl_file_path2)



