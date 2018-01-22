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
from woe.GridSearch import grid_search_lr_c as gs
from sklearn.svm import l1_min_c

def gen_sub_dataset():
    """
    生成数据集：
    只保留有用的特征
    """
    config_path = r'E:\Code\Python_ML_Code\cs_model\config\config_cs_daily_model_lr.csv'
    config = pd.read_csv(config_path)
    features_list = config[config.is_modelfeature==1].var_name
    features_list = list(features_list)
    features_list.append('raw_cs_cpd')
    features_list.append('target')
    features_list.append('contract_no')
    dataset_list = []

    for i in range(24):
        dataset_path = 'E:\\ScoreCard\\cs_model\\cs_m1_pos_model_daily\\raw_data\\dataset_split_by_rows\\' \
                       + 'm1_rsx_cs_unify_model_features_201705_daily_new_' \
                       + str(i+1) + '.csv'
        dataset = pd.read_csv(dataset_path)
        dataset['raw_cs_cpd'] = dataset['cs_cpd']
        dataset_list.append(dataset[features_list])
        print dataset.shape

    dataset_all = pd.DataFrame()
    dataset_all = dataset_all.append(dataset_list)
    dataset_all_path = 'E:\\ScoreCard\\cs_model\\cs_m1_pos_model_daily\\raw_data\\m1_rsx_cs_unify_model_features_201705_daily_new_sub.csv'
    dataset_all.to_csv(dataset_all_path,index=False)
    print dataset_all.shape
    print dataset.dtypes


def process_train_woe(infile_path=None,outfile_path=None,rst_path=None):
    print 'run into process_train_woe: \n',time.asctime(time.localtime(time.time()))
    config_path = r'E:\Code\Python_ML_Code\cs_model\config\config_cs_daily_model_lr.csv'
    data_path = infile_path
    cfg = config.config()
    cfg.load_file(config_path,data_path)

    # rst = []
    output = open(rst_path, 'rb')
    rst = pickle.load(output)
    output.close()

    exists_var_list = [rst[i].var_name for i in range(rst.__len__())]
    bin_var_list = [tmp for tmp in cfg.bin_var_list if tmp in list(cfg.dataset_train.columns) and tmp not in exists_var_list]

    for var in bin_var_list:
        # fill null
        cfg.dataset_train.loc[cfg.dataset_train[var].isnull(), (var)] = -1

    # change feature dtypes
    fp.change_feature_dtype(cfg.dataset_train, cfg.variable_type)

    # process woe transformation of continuous variables
    print 'process woe transformation of continuous variables: \n',time.asctime(time.localtime(time.time()))
    print 'cfg.global_bt',cfg.global_bt
    print 'cfg.global_gt', cfg.global_gt

    for var in bin_var_list:
        print var
        if rst.__len__()==0:
            pass
        else:
            output = open(rst_path, 'rb')
            rst = pickle.load(output)
            output.close()
            print 'load'
        rst.append(fp.proc_woe_continuous(cfg.dataset_train,var,cfg.global_bt,cfg.global_gt,cfg.min_sample,alpha=0.05))
        output = open(rst_path, 'wb')
        pickle.dump(rst,output)
        output.close()
        print 'dump'

    # process woe transformation of discrete variables
    print 'process woe transformation of discrete variables: \n',time.asctime(time.localtime(time.time()))
    for var in [tmp for tmp in cfg.discrete_var_list if tmp in list(cfg.dataset_train.columns) and tmp not in exists_var_list]:
        print var
        # fill null
        cfg.dataset_train.loc[cfg.dataset_train[var].isnull(), (var)] = 'missing'
        if rst.__len__()==0:
            pass
        else:
            output = open(rst_path, 'rb')
            rst = pickle.load(output)
            output.close()
            print 'load'
        rst.append(fp.proc_woe_discrete(cfg.dataset_train,var,cfg.global_bt,cfg.global_gt,cfg.min_sample,alpha=0.05))
        output = open(rst_path, 'wb')
        pickle.dump(rst,output)
        output.close()
        print 'dump'

    feature_detail = eval.eval_feature_detail(rst, outfile_path)
    return feature_detail,rst


def process_woe_trans(in_data_path=None,rst_path=None,out_path=None):
    print time.asctime(time.localtime(time.time())),'load config file'
    config_path = r'E:\Code\Python_ML_Code\cs_model\config\config_cs_daily_model.csv'
    data_path = in_data_path
    cfg = config.config()
    cfg.load_file(config_path, data_path)

    print time.asctime(time.localtime(time.time())),'fill na'
    dataset = pd.read_csv(in_data_path)

    print time.asctime(time.localtime(time.time())),'fill na continuous variables'
    for var in [tmp for tmp in cfg.bin_var_list if tmp in list(dataset.columns)]:
        # fill null
        dataset.loc[dataset[var].isnull(), (var)] = -1

    print time.asctime(time.localtime(time.time())),'fill na discrete variables'
    for var in [tmp for tmp in cfg.discrete_var_list if tmp in list(dataset.columns)]:
        # fill null
        dataset.loc[dataset[var].isnull(), (var)] = 'missing'

    print time.asctime(time.localtime(time.time())),'change feature dtypes'
    fp.change_feature_dtype(dataset, cfg.variable_type)

    print time.asctime(time.localtime(time.time())),'load woe rule'
    output = open(rst_path, 'rb')
    rst = pickle.load(output)
    output.close()

    # Training dataset Woe Transformation
    for r in rst:
        print 'woe trans:',r.var_name
        dataset[r.var_name] = fp.woe_trans(dataset[r.var_name], r)

    dataset.to_csv(out_path,index=False)
    print('%s\tSUCCESS EXPORT FILE: \n%s' %(time.asctime(time.localtime(time.time())),out_path))



def grid_search_lr_c(X_train,y_train,cs=[0.01],df_coef_path=False
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
    counter = 0
    for c in cs:
        print 'time: ',time.asctime(time.localtime(time.time())),'counter: ',counter, ' c: ',c
        clf_l1_LR.set_params(C=c)
        clf_l1_LR.fit(X_train, y_train)
        coefs_.append(clf_l1_LR.coef_.ravel().copy())

        proba = clf_l1_LR.predict_proba(X_train)[:,1]
        ks.append(compute_ks(proba,y_train))
        print 'ks:',ks[-1]
        counter += 1

    end = datetime.now()
    print end
    print("This took ", end - start)
    coef_cv_df = pd.DataFrame(coefs_,columns=X_train.columns)
    coef_cv_df['ks'] = ks
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


def grid_search_lr_c_main():
    print 'run into grid_search_lr_c_main:'
    print '--v8--'
    dataset_path = 'E:\\ScoreCard\\cs_model\\cs_m1_pos_model_daily\\gendata\\m1_cs_201705_daily_new_sub_woe_transed_train.csv'
    config_path = r'E:\Code\Python_ML_Code\cs_model\config\config_cs_daily_model.csv'
    df_coef_path = 'E:\\ScoreCard\\cs_model\\cs_m1_pos_model_daily\\eval\\cs_m1_pos_daily_coef_path_features_201705_v8.csv'

    pic_coefpath = 'E:\\ScoreCard\\cs_model\\cs_m1_pos_model_daily\\eval\\cs_m1_pos_daily_coef_path_rule_201705_v8.png'
    pic_performance = 'E:\\ScoreCard\\cs_model\\cs_m1_pos_model_daily\\eval\\cs_m1_pos_daily_performance_path_201705_v8.png'
    pic_coefpath_title = 'cs_m1_pos_daily_coef_path_rule_201705_v8'
    pic_performance_title = 'cs_m1_pos_daily_performance_path_201705_v8'

    dataset_train = pd.read_csv(dataset_path)

    cfg = pd.read_csv(config_path)
    candidate_var_list = cfg[cfg['is_modelfeature'] == 1]['var_name']

    b = [var for var in dataset_train.columns if sum(dataset_train[var].isnull()) == 0]
    candidate_var_list = list(set(candidate_var_list).intersection(set(b)))

    var_list_specfied = ['value_income_ratio'
                        ,'avg_rollseq'
                        ,'csfq'
                        ,'bptp_ratio'
                        ,'cert_4_inital'
                        ,'ptp'
                        ,'city'
                        ,'person_sex'
                        ,'over_due_value'
                        ,'most_contact_3m'
                        ,'recent_contact_day'
                        ,'rej_count'
                        ,'avg_days'
                        ,'max_cpd'
                        ,'finish_periods_ratio'
                        ,'due_periods_ratio'
                        ,'seq_delay_days']

    candidate_var_list = list(set(candidate_var_list).intersection(set(var_list_specfied)))
    print 'candidate_var_list:\n',candidate_var_list

    print 'change dtypes:float64 to float32'
    for var in candidate_var_list:
        dataset_train[var] = dataset_train[var].astype(np.float32)

    # candidate_var_list.append('raw_cs_cpd')

    # dataset_train['log_cs_cpd'] = np.log(dataset_train.raw_cs_cpd)
    # candidate_var_list.append('log_cs_cpd')

    dataset_train['trans_cs_cpd'] = np.log(dataset_train.raw_cs_cpd/31.0)- np.log(1-dataset_train.raw_cs_cpd/31.0)
    candidate_var_list.append('trans_cs_cpd')

    dataset_train['trans_cs_cpd2'] = np.log(dataset_train.raw_cs_cpd/31.0)- np.log(1-dataset_train.raw_cs_cpd/31.0)
    candidate_var_list.append('trans_cs_cpd2')

    X_train = dataset_train[candidate_var_list]
    y_train = dataset_train['target']
    # cs = l1_min_c(X_train, y_train, loss='log') * np.logspace(0, 6,50) # 6.1440502691447296e-08
    cs = np.logspace(-8, -2,30)
    print 'cs',cs
    c,ks = grid_search_lr_c(X_train,y_train,cs,df_coef_path,pic_coefpath_title,pic_coefpath
                               ,pic_performance_title,pic_performance)

    print 'pic_coefpath:\n',pic_coefpath
    print 'pic_performance:\n',pic_performance
    print 'ks performance on the c:'
    print c,ks

    return (c,ks)


def split_train_test_dataset():
    dataset_path = 'E:\\ScoreCard\\cs_model\\cs_m1_pos_model_daily\\gendata\\m1_cs_201705_daily_new_sub_woe_transed.csv'
    print 'load the full dataset:',dataset_path
    dataset = pd.read_csv(dataset_path)
    print 'suc load'
    dataset_train = dataset[:8000000]
    dataset_test = dataset[8000000:]

    dataset_train_path = 'E:\\ScoreCard\\cs_model\\cs_m1_pos_model_daily\\gendata\\m1_cs_201705_daily_new_sub_woe_transed_train.csv'
    print 'dump train dataset:',dataset_train_path
    dataset_train.to_csv(dataset_train_path,index=False)
    print 'done'

    dataset_test_path = 'E:\\ScoreCard\\cs_model\\cs_m1_pos_model_daily\\gendata\\m1_cs_201705_daily_new_sub_woe_transed_test.csv'
    print 'dump test dataset:',dataset_test_path
    dataset_test.to_csv(dataset_test_path,index=False)
    print 'done'



if __name__ == '__main__':
    # gen_sub_dataset()

    infile_path = 'E:\\ScoreCard\\cs_model\\cs_m1_pos_model_daily\\raw_data\\m1_rsx_cs_unify_model_features_201705_daily_new_sub.csv'
    outfile_path = 'E:\\ScoreCard\\cs_model\\cs_m1_pos_model_daily\\features_detail\\m1_cs_features_detail_201705_daily_new_sub.csv'
    rst_path = 'E:\\ScoreCard\\cs_model\\cs_m1_pos_model_daily\\gendata\\m1_cs_woe_rule_201705_daily_new_sub.pkl'
    # process_train_woe(infile_path,outfile_path,rst_path)

    out_path = 'E:\\ScoreCard\\cs_model\\cs_m1_pos_model_daily\\gendata\\m1_cs_201705_daily_new_sub_woe_transed.csv'
    # process_woe_trans(infile_path,rst_path,out_path)
    # split_train_test_dataset()
    grid_search_lr_c_main()