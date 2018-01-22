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


# proc woe transformation
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


def grid_search_lr_c_main(params):
    print 'run into grid_search_lr_c_main:'
    dataset_path = params['dataset_path']
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

    print 'candidate_var_list:\n',candidate_var_list

    print 'change dtypes:float64 to float32'
    for var in candidate_var_list:
        dataset_train[var] = dataset_train[var].astype(np.float32)

    X_train = dataset_train[candidate_var_list]
    y_train = dataset_train['target']

    cs = params['cs']
    print 'cs',cs
    c,ks = grid_search_lr_c(X_train,y_train,cs,df_coef_path,pic_coefpath_title,pic_coefpath
                            ,pic_performance_title,pic_performance)

    print 'pic_coefpath:\n',pic_coefpath
    print 'pic_performance:\n',pic_performance
    print 'ks performance on the c:'
    print c,ks

    return (c,ks)

if __name__ == '__main__':
    config_path = r'E:\Code\Python_ML_Code\cs_model\config\config_cs_model.csv'
    dataset_path = r'E:\ScoreCard\cs_model\cs_m1_pos_model\raw_data\m1_rsx_cs_unify_model_features_201710.csv'
    feature_detail,rst = process_train_woe(infile_path=dataset_path
                                           ,outfile_path=r'E:\ScoreCard\cs_model\cs_m1_pos_model\features_detail\cs_m1_pos_201710_features_detail.csv'
                                           ,rst_path=r'E:\ScoreCard\cs_model\cs_m1_pos_model\gendata\WOE_Rule\cs_m1_pos_woe_rule_201710.pkl')

    dataset_path = r'E:\ScoreCard\cs_model\cs_m1_pos_model\raw_data\m1_rsx_cs_unify_model_features_201711.csv'
    feature_detail,rst = process_train_woe(infile_path=dataset_path
                                           ,outfile_path=r'E:\ScoreCard\cs_model\cs_m1_pos_model\features_detail\cs_m1_pos_201711_features_detail.csv'
                                           ,rst_path=r'E:\ScoreCard\cs_model\cs_m1_pos_model\gendata\WOE_Rule\cs_m1_pos_woe_rule_201711.pkl')

    # rst_path=r'E:\ScoreCard\cs_model\cs_m2_pos_model\gendata\cs_m2_pos_woe_rule_912.pkl'
    # outfile_path=r'E:\ScoreCard\cs_model\cs_m2_pos_model\gendata\cs_m2_pos_912_woed.csv'
    # process_woe_trans(all_pos_path,rst_path,outfile_path)

    # print '###################################v1##############################################'
    # params = {}
    # params['dataset_path'] = r'E:\ScoreCard\cs_model\cs_m2_pos_model\gendata\cs_m2_pos_912_woed.csv'
    # params['config_path'] = r'E:\Code\Python_ML_Code\cs_model\config\config_cs_model_pos_m2.csv'
    # params['df_coef_path'] = r'E:\ScoreCard\cs_model\cs_m2_pos_model\eval\cs_m2_pos_912_v1.csv'
    #
    # params['pic_coefpath'] = r'E:\ScoreCard\cs_model\cs_m2_pos_model\eval\cs_m2_pos_coef_path_912_v1.png'
    # params['pic_performance'] = r'E:\ScoreCard\cs_model\cs_m2_pos_model\eval\cs_m2_pos_performance_path_912_v1.png'
    # params['pic_coefpath_title'] = 'cs_m2_pos_coef_path_912_v1'
    # params['pic_performance_title'] = 'cs_m2_pos_performance_path_912_v1'
    #
    # params['var_list_specfied'] = []
    # params['cs'] = np.logspace(-6, -1,30)
    # grid_search_lr_c_main(params)