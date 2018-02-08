# -*- coding:utf-8 -*-
__author__ = 'maomaochong'
import pandas as pd
import numpy as np
import pickle
from sklearn.linear_model import LogisticRegression
from woe.eval import  compute_ks
import woe.config as config

def fit_single_lr(dataset_path,config_path,var_list_specfied,out_model_path):
    dataset_train = pd.read_csv(dataset_path)
    cfg = pd.read_csv(config_path)
    candidate_var_list = cfg[cfg['is_modelfeature'] == 1]['var_name']

    b = [var for var in dataset_train.columns if sum(dataset_train[var].isnull()) == 0]
    candidate_var_list = list(set(candidate_var_list).intersection(set(b)))

    if var_list_specfied.__len__()>0:
        candidate_var_list = list(set(candidate_var_list).intersection(set(var_list_specfied)))

    print 'candidate_var_list length:\n',candidate_var_list.__len__()
    print 'candidate_var_list:\n',candidate_var_list

    print 'change dtypes:float64 to float32'
    for var in candidate_var_list:
        dataset_train[var] = dataset_train[var].astype(np.float32)

    X_train = dataset_train[dataset_train.target >=0][candidate_var_list]
    y_train = dataset_train[dataset_train.target >=0]['target']

    c = 0.008376776
    print 'c:',c
    # clf_lr_a = LogisticRegression(C=c, penalty='l1', tol=0.01,class_weight='balanced')
    clf_lr_a = LogisticRegression(C=c, penalty='l1', tol=0.01)

    clf_lr_a.fit(X_train, y_train)
    coefs = clf_lr_a.coef_.ravel().copy()

    proba = clf_lr_a.predict_proba(X_train)[:,1]
    ks = compute_ks(proba,y_train)

    model = {}
    model['clf'] = clf_lr_a
    model['features_list'] = candidate_var_list
    model['coefs'] = coefs
    model['ks'] = ks

    output = open(out_model_path, 'wb')
    pickle.dump(model,output)
    output.close()

    return model



woe_train_path3 = r'E:\work_file\mmt_application_card\gendata\mmt_application_model_train3_woed.csv'
woe_test_path3 = r'E:\work_file\mmt_application_card\gendata\mmt_application_model_test3_woed.csv'
config_path = r'E:\work_file\mmt_application_card\config\config_mmt_application_model.csv'
out_model_path = r'E:\work_file\mmt_application_card\LogisticRegression_Model\mmt_acard_target3.pkl'

var_list_specfied = ['pos_credit',
                     'rate_num_charcnt',
                     'rate_name_mincnt',
                     'callin_mem_max_cnt',
                     'name_avgcnt',
                     'dura_dep_type',
                     'rate_name_one_cnt',
                     'callout_mem_avg_duration',
                     'rate_num_dup_cnt',
                     'pos_on_time_pay_cnt',
                     'addr_active_call_cnt',
                     'ivsscore',
                     'total_mem_avg_duration',
                     'fim_nf_bci3',
                     'callout_mem_avg_cnt',
                     'llineal_contact_type',
                     'rate_num_sj_cnt',
                     'total_date_diff',
                     'callin_avg_duration',
                     'rate_name_maxcnt',
                     'dura_auth_page',
                     'al_m6_id_notbank_allnum',
                     'dura_bankcard_num',
                     'al_m3_id_notbank_allnum',
                     'rate_name_dup_cnt',
                     'addr_valid_call_cnt',
                     'cpl_cps_level',
                     'phonecall_total_cnt',
                     'miss_mem_cnt',
                     'dura_home_page',
                     'total_duration',
                     'rate_num_gh_cnt',
                     'callout_mem_max_cnt',
                     'rate_num_cnt',
                     'al_m3_cell_notbank_allnum',
                     'al_m6_cell_notbank_allnum',
                     'total_avg_duration',
                     'age',
                     'pos_cur_banlance',
                     'rate_name_avgcnt',
                     'data_status',
                     'pos_in_time_pay_cnt',
                     'pos_dd_fail_cnt',
                     'al_m12_id_notbank_allnum',
                     'dura_cert_pic2',
                     'rate_without_areacode_cnt',
                     'miss_cnt',
                     'gender',
                     'rate_spchar_cnt',
                     'zm_name_ret_code',
                     'addr_active_call_rate',
                     'non_addr_mem_cnt',
                     'dura_cert_wg',
                     'pos_total_delay_day_cnt',
                     'qualification',
                     'td_decision',
                     'resmsg']

fit_single_lr(woe_train_path3,config_path,var_list_specfied,out_model_path)

config_path = r'E:\Code\Python_ML_Code\cs_model\config\config_cs_model.csv'
cfg = config.config()
cfg.load_file(config_path, woe_test_path3)

for var in [tmp for tmp in cfg.bin_var_list if tmp in list(cfg.dataset_train.columns)]:
    # fill null
    cfg.dataset_train.loc[cfg.dataset_train[var].isnull(), (var)] = 0

for var in [tmp for tmp in cfg.discrete_var_list if tmp in list(cfg.dataset_train.columns)]:
    # fill null
    cfg.dataset_train.loc[cfg.dataset_train[var].isnull(), (var)] = 0

output = open(out_model_path, 'rb')
clf_model = pickle.load(output)
output.close()

clf = clf_model['clf']
X_test = cfg.dataset_train[clf_model['features_list']]
y_test = cfg.dataset_train['target']

y_hat = clf.predict_proba(X_test)[:,1]
ks = compute_ks(y_hat,y_test)
