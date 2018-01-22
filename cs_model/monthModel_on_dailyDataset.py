# -*- coding:utf-8 -*-
__author__ = 'maomaochong'
import pandas as pd
import woe.config as config
import woe.feature_process as fp
import numpy as np
import pickle
from woe.eval import  compute_ks


def process_woe_trans(in_data_path=None,rst_path=None):
    config_path = r'E:\Code\Python_ML_Code\cs_model\config\config_cs_model_201705.csv'
    data_path = in_data_path
    cfg = config.config()
    cfg.load_file(config_path, data_path)

    cfg.dataset_train = cfg.dataset_train.rename(columns={'cs_cpd':'cpd'}) # rename
    # dataset['raw_cs_cpd'] = dataset['cs_cpd']

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

    return cfg.dataset_train


if __name__ == '__main__':
    # load month model
    clf_path = r'E:\ScoreCard\cs_model\cs_m1_pos_model\gendata\LogisticRegression_Model\cs_m1_pos_clf_201705.pkl'
    output = open(clf_path, 'rb')
    clf = pickle.load(output)
    output.close()

    # 定义数据结构
    model_ks_dict = {}
    for i in range(30):
        model_ks_dict[i+1] = {}
        model_ks_dict[i+1]['y_train'] = []
        model_ks_dict[i+1]['y_hat'] = []

    # load raw dataset
    for i in range(24):
        # dataset
        dataset_path = 'E:\\ScoreCard\\cs_model\\cs_m1_pos_model_daily\\raw_data\\dataset_split_by_rows\\m1_rsx_cs_unify_model_features_201705_daily_new_' \
                       +str(i+1)+'.csv'
        print 'running:',dataset_path
        # woe transformation
        rst_path = 'E:\\ScoreCard\\cs_model\\cs_m1_pos_model\\gendata\\WOE_Rule\\cs_m1_pos_woe_rule_201705.pkl'
        dataset_woe_transed = process_woe_trans(in_data_path=dataset_path,rst_path=rst_path)

        # dataset = pd.read_csv(dataset_path)
        cols_to_keep = clf['features_name']
        cols_to_keep.append('target')
        cols_to_keep.append('cpd')
        # dataset_woe_transed = dataset_woe_transed[cols_to_keep]

        # cpd
        for j in range(30):
            print 'cpd:',j
            subset = dataset_woe_transed[dataset_woe_transed['cpd']==(j+1)]

            X_train = subset[clf['features_name'][:18]].values
            y_train = subset['target'].values
            model_ks_dict[j+1]['y_train'].extend(y_train)

            proba = clf['classifier'].predict_proba(X_train)[:,1]
            # ks = compute_ks(proba,y_train)
            model_ks_dict[j+1]['y_hat'].extend(proba)


    for j in range(30):
        ks = compute_ks(np.array(model_ks_dict[j+1]['y_hat']),np.array(model_ks_dict[j+1]['y_train']))
        sample_cnt = model_ks_dict[j+1]['y_train'].__len__()
        bad_sample_cnt = sum(model_ks_dict[j+1]['y_train'])
        bad_sample_rate = bad_sample_cnt*1.0/sample_cnt
        print('cs_cpd:%s\tsample_cnt:%s\tbad_sample_cnt:%s\tbad_sample_rate:%s\tks:%s'
              % (str(j+1),str(sample_cnt),str(bad_sample_cnt),str(bad_sample_rate),str(ks)))

    v_cpd = []
    v_y_train = []
    v_y_proba = []
    for j in range(30):
        v_cpd.extend([j]*model_ks_dict[j+1]['y_train'].__len__())
        v_y_train.extend(model_ks_dict[j+1]['y_train'])
        v_y_proba.extend(model_ks_dict[j+1]['y_hat'])

    ks_test = pd.DataFrame()
    ks_test['cpd'] = pd.Series(v_cpd)
    ks_test['y_train'] = pd.Series(v_y_train)
    ks_test['y_proba'] = pd.Series(v_y_proba)
    ks_test.to_csv('E:\ks_test_daily.csv',index=False)