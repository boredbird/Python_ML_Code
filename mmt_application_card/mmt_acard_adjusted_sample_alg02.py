# -*- coding:utf-8 -*-
__author__ = 'maomaochong'
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import woe.feature_process as fp
import woe.eval as eval
import woe.config as config
import pickle
import time
from datetime import datetime
from sklearn.linear_model import LogisticRegression
from woe.eval import  compute_ks

# train woe rule
"""
训练WOE规则
"""
def process_train_woe(infile_path=None,outfile_path=None,rst_path=None,config_path=None):
    print 'run into process_train_woe: \n',time.asctime(time.localtime(time.time()))
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


# proc woe transformation
"""
进行WOE转换
"""
def process_woe_trans(in_data_path=None,rst_path=None,out_path=None,config_path=None):
    cfg = config.config()
    cfg.load_file(config_path, in_data_path)

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


def grid_search_lr_c_validation(X_train,y_train,validation_dataset_list,cs=[0.01],df_coef_path=False
                                ,pic_coefpath_title='Logistic Regression Path',pic_coefpath=False
                                ,pic_performance_title='Logistic Regression Performance',pic_performance=False):
    """
    grid search optimal hyper parameters c with the best ks performance
    :param X_train: features dataframe
    :param y_train: target
    :param cs: list of c value
    :param df_coef_path: the file path for logistic regression coefficient dataframe
    :param pic_coefpath_title: the pic title for coefficient path picture
    :param pic_coefpath: the file path for coefficient path picture
    :param pic_performance_title: the pic title for ks performance picture
    :param pic_performance: the file path for ks performance picture
    :return: a tuple of c and ks value with the best ks performance
    """
    # init a LogisticRegression model
    clf_l1_LR = LogisticRegression(C=0.1, penalty='l1', tol=0.01,class_weight='balanced')

    print("Computing regularization path ...")
    start = datetime.now()
    print start
    coefs_ = []
    ks = []
    ks_validation1 = []
    ks_validation2 = []
    counter = 0
    for c in cs:
        print 'time: ',time.asctime(time.localtime(time.time())),'counter: ',counter, ' c: ',c
        clf_l1_LR.set_params(C=c)
        clf_l1_LR.fit(X_train, y_train)
        coefs_.append(clf_l1_LR.coef_.ravel().copy())

        proba = clf_l1_LR.predict_proba(X_train)[:,1]
        validation_proba1 = clf_l1_LR.predict_proba(validation_dataset_list[0][X_train.columns])[:,1]

        ks.append(compute_ks(proba,y_train))
        ks_validation1.append(compute_ks(validation_proba1,validation_dataset_list[0]['target']))

        print 'ks:\t',ks[-1],'ks_validation1:\t',ks_validation1[-1]
        counter += 1

    end = datetime.now()
    print end
    print("This took ", end - start)
    coef_cv_df = pd.DataFrame(coefs_,columns=X_train.columns)
    coef_cv_df['ks'] = ks
    coef_cv_df['ks_validation1'] = ks_validation1
    coef_cv_df['c'] = cs


    if df_coef_path:
        file_name = df_coef_path if isinstance(df_coef_path, str) else None
        coef_cv_df.to_csv(file_name)

    coefs_ = np.array(coefs_)

    fig1 = plt.figure('fig1')
    plt.plot(np.log10(cs), coefs_)
    ymin, ymax = plt.ylim()
    plt.xlabel('log(C)')
    plt.ylabel('Coefficients')
    plt.title(pic_coefpath_title)
    plt.axis('tight')
    if pic_coefpath:
        file_name = pic_coefpath if isinstance(pic_coefpath, str) else None
        plt.savefig(file_name)
        plt.close()
    else:
        pass
        # plt.show()
        # plt.close()

    fig2 = plt.figure('fig2')
    plt.plot(np.log10(cs), ks)
    plt.xlabel('log(C)')
    plt.ylabel('ks score')
    plt.title(pic_performance_title)
    plt.axis('tight')
    if pic_performance:
        file_name = pic_performance if isinstance(pic_performance, str) else None
        plt.savefig(file_name)
        plt.close()
    else:
        pass
        # plt.show()
        # plt.close()

    flag = coefs_<0
    if np.array(ks)[flag.sum(axis=1) == 0].__len__()>0:
        idx = np.array(ks)[flag.sum(axis=1) == 0].argmax()
    else:
        idx = np.array(ks).argmax()

    return (cs[idx],ks[idx])


def grid_search_lr_c_main(params):
    print 'run into grid_search_lr_c_main:'
    dataset_path = params['dataset_path']
    validation_path = params['validation_path']
    config_path = params['config_path']
    df_coef_path = params['df_coef_path']
    pic_coefpath = params['pic_coefpath']
    pic_performance = params['pic_performance']
    pic_coefpath_title = params['pic_coefpath_title']
    pic_performance_title = params['pic_performance_title']

    dataset_train = pd.read_csv(dataset_path)
    cfg = pd.read_csv(config_path)
    candidate_var_list = cfg[cfg['is_modelfeature'] == 1]['var_name']

    b = [var for var in dataset_train.columns if sum(dataset_train[var].isnull()) == 0]
    candidate_var_list = list(set(candidate_var_list).intersection(set(b)))

    var_list_specfied = params['var_list_specfied']
    if var_list_specfied.__len__()>0:
        candidate_var_list = list(set(candidate_var_list).intersection(set(var_list_specfied)))

    print 'candidate_var_list length:\n',candidate_var_list.__len__()
    print 'candidate_var_list:\n',candidate_var_list

    print 'change dtypes:float64 to float32'
    for var in candidate_var_list:
        dataset_train[var] = dataset_train[var].astype(np.float32)

    X_train = dataset_train[dataset_train.target >=0][candidate_var_list]
    y_train = dataset_train[dataset_train.target >=0]['target']

    validation_cols_keep = [var for var in candidate_var_list]
    validation_cols_keep.append('target')
    validation_dataset_list = []

    validation_dataset = pd.read_csv(validation_path)
    # fillna
    for var in candidate_var_list:
        validation_dataset.loc[validation_dataset[var].isnull(), (var)] = 0
    validation_dataset_list.append(validation_dataset[validation_cols_keep])

    cs = params['cs']
    print 'cs',cs
    # c,ks = grid_search_lr_c(X_train,y_train,cs,df_coef_path,pic_coefpath_title,pic_coefpath
    #                         ,pic_performance_title,pic_performance)


    c,ks = grid_search_lr_c_validation(X_train,y_train,validation_dataset_list,cs,df_coef_path,pic_coefpath_title,pic_coefpath
                                       ,pic_performance_title,pic_performance)
    print 'pic_coefpath:\n',pic_coefpath
    print 'pic_performance:\n',pic_performance
    print 'ks performance on the c:'
    print c,ks

    return (c,ks)


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

    c = 0.01
    print 'c:',c
    clf_lr_a = LogisticRegression(C=c, penalty='l1', tol=0.01,class_weight='balanced')

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

def proc_validattion(dataset_path,config_path,model_path):
    print '####PROC VALIDATION#####'
    print 'dataset_path:\n',dataset_path
    print 'config_path:\n',config_path
    print 'model_path:\n',model_path
    #fillna
    config_path = r'E:\Code\Python_ML_Code\cs_model\config\config_cs_model.csv'
    cfg = config.config()
    cfg.load_file(config_path, dataset_path)

    for var in [tmp for tmp in cfg.bin_var_list if tmp in list(cfg.dataset_train.columns)]:
        # fill null
        cfg.dataset_train.loc[cfg.dataset_train[var].isnull(), (var)] = 0

    for var in [tmp for tmp in cfg.discrete_var_list if tmp in list(cfg.dataset_train.columns)]:
        # fill null
        cfg.dataset_train.loc[cfg.dataset_train[var].isnull(), (var)] = 0

    output = open(model_path, 'rb')
    clf_model = pickle.load(output)
    output.close()

    clf = clf_model['clf']
    X_test = cfg.dataset_train[clf_model['features_list']]
    y_test = cfg.dataset_train['target']

    y_hat = clf.predict_proba(X_test)[:,1]
    ks = compute_ks(y_hat,y_test)
    print 'global_bt:',cfg.global_bt
    print 'global_gt:', cfg.global_gt
    print 'ks:',ks
    return ks


def proc_cor_eval(dataset_path,config_path,var_list_specfied,out_file_path):
    dataset = pd.read_csv(dataset_path)
    cfg = pd.read_csv(config_path)
    candidate_var_list = cfg[cfg['is_modelfeature'] == 1]['var_name']

    b = [var for var in dataset.columns if sum(dataset[var].isnull()) == 0]
    candidate_var_list = list(set(candidate_var_list).intersection(set(b)))

    if var_list_specfied.__len__()>0:
        candidate_var_list = list(set(candidate_var_list).intersection(set(var_list_specfied)))

    print 'candidate_var_list length:\n',candidate_var_list.__len__()
    print 'candidate_var_list:\n',candidate_var_list

    cor = np.corrcoef(dataset[candidate_var_list].values,rowvar=0)
    pd.DataFrame(cor).to_csv(out_file_path,index=False)


if __name__ == '__main__':
    dataset_path = r'E:\work_file\mmt_application_card\raw_data\mmt_application_model_feature_f.csv'
    dataset_train_path1 = r'E:\work_file\mmt_application_card\sample_weight_adjustment\weight012\raw_data\mmt_acard_feature_train1.csv'
    dataset_train_path2 = r'E:\work_file\mmt_application_card\sample_weight_adjustment\weight012\raw_data\mmt_acard_feature_train2.csv'
    dataset_train_path3 = r'E:\work_file\mmt_application_card\sample_weight_adjustment\weight012\raw_data\mmt_acard_feature_train3.csv'

    dataset_test_path1 = r'E:\work_file\mmt_application_card\raw_data\mmt_application_model_feature_ftest1.csv'
    dataset_test_path2 = r'E:\work_file\mmt_application_card\raw_data\mmt_application_model_feature_ftest2.csv'
    dataset_test_path3 = r'E:\work_file\mmt_application_card\raw_data\mmt_application_model_feature_ftest3.csv'

    config_path = r'E:\work_file\mmt_application_card\config\config_mmt_application_model.csv'

    feature_detail_path1 = r'E:\work_file\mmt_application_card\sample_weight_adjustment\weight012\mmt_acard_reweighted_features_detail1.csv'
    feature_detail_path2 = r'E:\work_file\mmt_application_card\sample_weight_adjustment\weight012\mmt_acard_reweighted_features_detail2.csv'
    feature_detail_path3 = r'E:\work_file\mmt_application_card\sample_weight_adjustment\weight012\mmt_acard_reweighted_features_detail3.csv'
    rst_pkl_path1 = r'E:\work_file\mmt_application_card\sample_weight_adjustment\weight012\mmt_acard_reweighted_woe_rule1.pkl'
    rst_pkl_path2 = r'E:\work_file\mmt_application_card\sample_weight_adjustment\weight012\mmt_acard_reweighted_woe_rule2.pkl'
    rst_pkl_path3 = r'E:\work_file\mmt_application_card\sample_weight_adjustment\weight012\mmt_acard_reweighted_woe_rule3.pkl'
    # feature_detail,rst = process_train_woe(infile_path=dataset_train_path1
    #                                        ,outfile_path=feature_detail_path1
    #                                        ,rst_path=rst_pkl_path1
    #                                        ,config_path=config_path)
    # feature_detail,rst = process_train_woe(infile_path=dataset_train_path2
    #                                        ,outfile_path=feature_detail_path2
    #                                        ,rst_path=rst_pkl_path2
    #                                        ,config_path=config_path)
    # feature_detail,rst = process_train_woe(infile_path=dataset_train_path3
    #                                        ,outfile_path=feature_detail_path3
    #                                        ,rst_path=rst_pkl_path3
    #                                        ,config_path=config_path)

    woe_train_path1 = r'E:\work_file\mmt_application_card\sample_weight_adjustment\weight012\mmt_application_model_train1_woed.csv'
    # process_woe_trans(dataset_train_path1,rst_pkl_path1,woe_train_path1,config_path)
    woe_test_path1 = r'E:\work_file\mmt_application_card\sample_weight_adjustment\weight012\mmt_application_model_test1_woed.csv'
    # process_woe_trans(dataset_test_path1,rst_pkl_path1,woe_test_path1,config_path)

    woe_train_path2 = r'E:\work_file\mmt_application_card\sample_weight_adjustment\weight012\mmt_application_model_train2_woed.csv'
    # process_woe_trans(dataset_train_path2,rst_pkl_path2,woe_train_path2,config_path)
    woe_test_path2 = r'E:\work_file\mmt_application_card\sample_weight_adjustment\weight012\mmt_application_model_test2_woed.csv'
    # process_woe_trans(dataset_test_path2,rst_pkl_path2,woe_test_path2,config_path)

    woe_train_path3 = r'E:\work_file\mmt_application_card\sample_weight_adjustment\weight012\mmt_application_model_train3_woed.csv'
    # process_woe_trans(dataset_train_path3,rst_pkl_path3,woe_train_path3,config_path)
    woe_test_path3 = r'E:\work_file\mmt_application_card\sample_weight_adjustment\weight012\mmt_application_model_test3_woed.csv'
    # process_woe_trans(dataset_test_path3,rst_pkl_path3,woe_test_path3,config_path)

    df_cor_path1 = r'E:\work_file\mmt_application_card\sample_weight_adjustment\weight012\mmt_application_model_train1_woed_df_cor.csv'
    df_cor_path2 = r'E:\work_file\mmt_application_card\sample_weight_adjustment\weight012\mmt_application_model_train2_woed_df_cor.csv'
    df_cor_path3 = r'E:\work_file\mmt_application_card\sample_weight_adjustment\weight012\mmt_application_model_train3_woed_df_cor.csv'

    # proc_cor_eval(woe_train_path1,config_path,var_list_specfied=[],out_file_path=df_cor_path1)
    # proc_cor_eval(woe_train_path2,config_path,var_list_specfied=[],out_file_path=df_cor_path2)
    # proc_cor_eval(woe_train_path3,config_path,var_list_specfied=[],out_file_path=df_cor_path3)

    print '###################################target1_v3##############################################'
    params1 = {}
    params1['dataset_path'] = woe_train_path1
    params1['validation_path'] = woe_test_path1
    params1['config_path'] = config_path

    params1['df_coef_path'] = r'E:\work_file\mmt_application_card\sample_weight_adjustment\weight012\mmt_application_model_df_coef_target1_v3.csv'
    params1['pic_coefpath'] = r'E:\work_file\mmt_application_card\sample_weight_adjustment\weight012\mmt_application_model_coefpath_target1_v3.png'
    params1['pic_performance'] = r'E:\work_file\mmt_application_card\sample_weight_adjustment\weight012\mmt_application_model_performance_path_target1_v3.png'
    params1['pic_coefpath_title'] = 'mmt_application_model_coefpath_target1_v3'
    params1['pic_performance_title'] = 'mmt_application_model_performance_path_target1_v3'

    params1['var_list_specfied'] = ['rate_num_charcnt',
                                    'al_m3_cell_notbank_allnum',
                                    'rate_num_sj_cnt',
                                    'dura_dep_type',
                                    'rate_name_one_cnt',
                                    'pos_finished_periods_cnt',
                                    'callout_mem_avg_duration',
                                    'total_cnt',
                                    'rate_num_dup_cnt',
                                    'addr_active_call_cnt',
                                    'ivsscore',
                                    'emp_province_name',
                                    'shop_status_pos',
                                    'dd5_pos',
                                    'llineal_contact_type',
                                    'id_prov_name',
                                    'pos_cur_banlance',
                                    'rate_name_maxcnt',
                                    'dura_auth_page',
                                    'al_m6_id_notbank_allnum',
                                    'rate_name_avgcnt',
                                    'rate_name_dup_cnt',
                                    'pos_credit',
                                    'fill_mob_phone_period',
                                    'dd7_pos',
                                    'dura_home_page',
                                    'scorebank',
                                    'rate_num_gh_cnt',
                                    'rate_num_cnt',
                                    'dura_emp_name',
                                    'rate_name_mincnt',
                                    'total_avg_duration',
                                    'callin_avg_duration',
                                    'data_status',
                                    'callout_mem_avg_cnt',
                                    'pos_in_time_pay_cnt',
                                    'pos_dd_fail_cnt',
                                    'day_week',
                                    'al_m12_id_notbank_allnum',
                                    'rate_without_areacode_cnt',
                                    'age',
                                    'rate_spchar_cnt',
                                    'zm_name_ret_code',
                                    'dd6_pos',
                                    'dura_cert_wg',
                                    'pos_total_delay_day_cnt',
                                    'zm_bank_no_ret_code',
                                    'td_decision',
                                    'name_avgcnt']
    params1['cs'] = np.logspace(-4, -1,40)
    for key,value in params1.items():
        print key,': ',value
    grid_search_lr_c_main(params1)

    print '###################################target2_v3##############################################'
    params2 = {}
    params2['dataset_path'] = woe_train_path2
    params2['validation_path'] = woe_test_path2
    params2['config_path'] = config_path

    params2['df_coef_path'] = r'E:\work_file\mmt_application_card\sample_weight_adjustment\weight012\mmt_application_model_df_coef_target2_v3.csv'
    params2['pic_coefpath'] = r'E:\work_file\mmt_application_card\sample_weight_adjustment\weight012\mmt_application_model_coefpath_target2_v3.png'
    params2['pic_performance'] = r'E:\work_file\mmt_application_card\sample_weight_adjustment\weight012\mmt_application_model_performance_path_target2_v3.png'
    params2['pic_coefpath_title'] = 'mmt_application_model_coefpath_target2_v3'
    params2['pic_performance_title'] = 'mmt_application_model_performance_path_target2_v3'

    params2['var_list_specfied'] = ['rate_num_charcnt',
                                    'rate_num_sj_cnt',
                                    'rate_ex_short_cnt',
                                    'rate_name_one_cnt',
                                    'fill_mob_phone_period',
                                    'callout_mem_avg_duration',
                                    'rate_num_dup_cnt',
                                    'pos_on_time_pay_cnt',
                                    'addr_active_call_cnt',
                                    'ivsscore',
                                    'emp_province_name',
                                    'non_addr_mem_rate',
                                    'dd5_pos',
                                    'llineal_contact_type',
                                    'id_prov_name',
                                    'num_gh_cnt',
                                    'pos_cur_banlance',
                                    'rate_name_num_dup_cnt',
                                    'rate_name_maxcnt',
                                    'dura_auth_page',
                                    'al_m6_id_notbank_allnum',
                                    'dura_bankcard_num',
                                    'rate_name_avgcnt',
                                    'pos_credit',
                                    'miss_mem_cnt',
                                    'dd7_pos',
                                    'total_duration',
                                    'dura_office_tel',
                                    'rate_num_cnt',
                                    'dura_emp_name',
                                    'rate_name_mincnt',
                                    'al_m6_cell_notbank_allnum',
                                    'total_avg_duration',
                                    'name_num_dup_cnt',
                                    'name_dup_cnt',
                                    'callin_avg_duration',
                                    'data_status',
                                    'callout_mem_avg_cnt',
                                    'pos_in_time_pay_cnt',
                                    'pos_dd_fail_cnt',
                                    'al_m12_id_notbank_allnum',
                                    'callout_date_diff',
                                    'day_week',
                                    'miss_cnt',
                                    'gender',
                                    'age',
                                    'rate_without_areacode_cnt',
                                    'name_one_cnt',
                                    'num_sj_cnt',
                                    'addr_active_call_rate',
                                    'dura_bindphone',
                                    'dd6_pos',
                                    'non_addr_mem_cnt',
                                    'pos_total_delay_day_cnt',
                                    'callin_cnt',
                                    'name_avgcnt']
    params2['cs'] = np.logspace(-4, -1,40)
    for key,value in params2.items():
        print key,': ',value
    grid_search_lr_c_main(params2)

    print '###################################target3_v3##############################################'
    params3 = {}
    params3['dataset_path'] = woe_train_path3
    params3['validation_path'] = woe_test_path3
    params3['config_path'] = config_path

    params3['df_coef_path'] = r'E:\work_file\mmt_application_card\sample_weight_adjustment\weight012\mmt_application_model_df_coef_target3_v3.csv'
    params3['pic_coefpath'] = r'E:\work_file\mmt_application_card\sample_weight_adjustment\weight012\mmt_application_model_coefpath_target3_v3.png'
    params3['pic_performance'] = r'E:\work_file\mmt_application_card\sample_weight_adjustment\weight012\mmt_application_model_performance_path_target3_v3.png'
    params3['pic_coefpath_title'] = 'mmt_application_model_coefpath_target3_v3'
    params3['pic_performance_title'] = 'mmt_application_model_performance_path_target3_v3'

    params3['var_list_specfied'] = ['rate_name_dup_cnt',
                                    'pos_credit',
                                    'fill_mob_phone_period',
                                    'al_m3_cell_notbank_allnum',
                                    'pos_cur_banlance',
                                    'data_status',
                                    'callout_mem_avg_duration',
                                    'dd6_pos',
                                    'td_decision',
                                    'resmsg',
                                    'rate_num_sj_cnt',
                                    'pos_finished_periods_cnt',
                                    'addr_active_call_cnt',
                                    'emp_province_name',
                                    'al_m6_id_notbank_allnum',
                                    'pos_total_delay_day_cnt',
                                    'rate_spchar_cnt',
                                    'name_avgcnt',
                                    'rate_num_dup_cnt',
                                    'pos_on_time_pay_cnt',
                                    'llineal_contact_type',
                                    'callin_mem_max_cnt',
                                    'rate_name_maxcnt',
                                    'rate_name_avgcnt',
                                    'rate_name_mincnt',
                                    'pos_dd_fail_cnt',
                                    'al_m12_id_notbank_allnum',
                                    'zm_name_ret_code',
                                    'callout_mem_max_cnt',
                                    'rate_name_one_cnt',
                                    'total_mem_avg_cnt',
                                    'ivsscore',
                                    'id_prov_name',
                                    'dura_auth_page',
                                    'rate_num_gh_cnt',
                                    'dura_emp_name',
                                    'callout_mem_avg_cnt',
                                    'pos_in_time_pay_cnt',
                                    'rate_num_cnt',
                                    'age']
    params3['cs'] = np.logspace(-4, -1,40)
    for key,value in params3.items():
        print key,': ',value
    grid_search_lr_c_main(params3)
    
    # fit_single_lr(dataset_path=params['dataset_path']
    #               ,config_path=params['config_path']
    #               ,var_list_specfied=params['var_list_specfied']
    #               ,out_model_path=r'E:\ScoreCard\cs_model\cs_m2_xcash_model\LogisticRegression_Model\cs_m2_xcash_clf_2017910.pkl')
    #
    # proc_validattion(dataset_path=r'E:\ScoreCard\cs_model\cs_m2_xcash_model\gendata\cs_m2_xcash_woe_rule_910_f0308_woed.csv'
    #                  ,config_path=params['config_path']
    #                  ,model_path=r'E:\ScoreCard\cs_model\cs_m2_xcash_model\LogisticRegression_Model\cs_m2_xcash_clf_2017910.pkl')
    #
    # proc_validattion(dataset_path=r'E:\ScoreCard\cs_model\cs_m2_xcash_model\gendata\cs_m2_xcash_woe_rule_910_f1112_woed.csv'
    #                  ,config_path=params['config_path']
    #                  ,model_path=r'E:\ScoreCard\cs_model\cs_m2_xcash_model\LogisticRegression_Model\cs_m2_xcash_clf_2017910.pkl')
