# -*- coding:utf-8 -*-
"""
根据缩小范围后的c值，对应的模型表现，
进一步分别确定每个模型的入模变量与最优的c值，并保存对应的模型文件
"""
__author__ = 'maomaochong'
import woe.GridSearch as gs
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from datetime import datetime
from woe.eval import  compute_ks
import time
import pickle

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

i = 8
print '[START]',time.asctime(time.localtime(time.time()))
dataset_path = 'E:\\ScoreCard\\cs_model\\gendata\\' + dataset_path_list[i]
df_coef_path = 'E:\\ScoreCard\\cs_model\\eval\\' + df_coef_path_list[i]
pic_coefpath_title = 'cs_m1_pos_coef_path_rule_20170'+str(i+1)+'_features_20170'+str(i+1)
pic_coefpath = 'E:\\ScoreCard\\cs_model\\eval\\' + 'cs_m1_pos_coef_path_rule_20170'+str(i+1) \
               +'_features_20170'+str(i+1)+'.png'
pic_performance_title = 'cs_m1_pos_performance_path_rule_20170'+str(i+1)+'_features_20170'+str(i+1)
pic_performance = 'E:\\ScoreCard\\cs_model\\eval\\' + 'cs_m1_pos_performance_path_rule_20170'+str(i+1) \
                  +'_features_20170'+str(i+1)+'.png'
config_path = 'E:\\Code\\Python_ML_Code\\cs_model\\config\\config_cs_model_20170'+str(i+1)+'.csv'

dataset_train = pd.read_csv(dataset_path)
# dataset_train = dataset_train.loc[:50000,]

cfg = pd.read_csv(config_path)
candidate_var_list = cfg[cfg['is_modelfeature'] == 1]['var_name']

b = [var for var in dataset_train.columns if sum(dataset_train[var].isnull()) == 0]
candidate_var_list = list(set(candidate_var_list).intersection(set(b)))

X_train = dataset_train[candidate_var_list]
y_train = dataset_train['target']

c,ks = gs.grid_search_lr_c(X_train,y_train,df_coef_path,pic_coefpath_title,pic_coefpath
                           ,pic_performance_title,pic_performance)



"""
Search for optimal hyper parametric C in LogisticRegression
"""
# cs = np.logspace(-5, -2,50)
# c,ks = grid_search_lr_c(X_train,y_train,cs,df_coef_path,pic_coefpath_title,pic_coefpath
#                            ,pic_performance_title,pic_performance)


# def grid_search_lr_c(X_train,y_train,cs=[0.1],df_coef_path=False
#                      ,pic_coefpath_title='Logistic Regression Path',pic_coefpath=False
#                      ,pic_performance_title='Logistic Regression Performance',pic_performance=False):
#     """
#     grid search optimal hyper parameters c with the best ks performance
#     :param X_train: features dataframe
#     :param y_train: target
#     :param cs: list of c value
#     :param df_coef_path: the file path for logistic regression coefficient dataframe
#     :param pic_coefpath_title: the pic title for coefficient path picture
#     :param pic_coefpath: the file path for coefficient path picture
#     :param pic_performance_title: the pic title for ks performance picture
#     :param pic_performance: the file path for ks performance picture
#     :return: a tuple of c and ks value with the best ks performance
#     """
#     # init a LogisticRegression model
#     clf_l1_LR = LogisticRegression(C=0.1, penalty='l1', tol=0.01,class_weight='balanced')
#     # cs = l1_min_c(X_train, y_train, loss='log') * np.logspace(0, 6,50)
#     print("Computing regularization path ...")
#     start = datetime.now()
#     print start
#     coefs_ = []
#     ks = []
#     counter = 0
#     for c in cs:
#         print 'counter: ',counter,' time: ',time.asctime(time.localtime(time.time())),' c: ',c
#         clf_l1_LR.set_params(C=c)
#         clf_l1_LR.fit(X_train, y_train)
#         coefs_.append(clf_l1_LR.coef_.ravel().copy())
#
#         proba = clf_l1_LR.predict_proba(X_train)[:,1]
#         ks.append(compute_ks(proba,y_train))
#         counter += 1
#
#     end = datetime.now()
#     print end
#     print("This took ", end - start)
#     coef_cv_df = pd.DataFrame(coefs_,columns=X_train.columns)
#     coef_cv_df['ks'] = ks
#     coef_cv_df['c'] = cs
#
#     if df_coef_path:
#         file_name = df_coef_path if isinstance(df_coef_path, str) else None
#         coef_cv_df.to_csv(file_name)
#
#     coefs_ = np.array(coefs_)
#
#     fig1 = plt.figure('fig1')
#     plt.plot(np.log10(cs), coefs_)
#     ymin, ymax = plt.ylim()
#     plt.xlabel('log(C)')
#     plt.ylabel('Coefficients')
#     plt.title(pic_coefpath_title)
#     plt.axis('tight')
#     if pic_coefpath:
#         file_name = pic_coefpath if isinstance(pic_coefpath, str) else None
#         plt.savefig(file_name)
#         plt.close()
#     else:
#         pass
#         # plt.show()
#         # plt.close()
#
#     fig2 = plt.figure('fig2')
#     plt.plot(np.log10(cs), ks)
#     plt.xlabel('log(C)')
#     plt.ylabel('ks score')
#     plt.title(pic_performance_title)
#     plt.axis('tight')
#     if pic_performance:
#         file_name = pic_performance if isinstance(pic_performance, str) else None
#         plt.savefig(file_name)
#         plt.close()
#     else:
#         pass
#         # plt.show()
#         # plt.close()
#
#     flag = coefs_<0
#     if np.array(ks)[flag.sum(axis=1) == 0].__len__()>0:
#         idx = np.array(ks)[flag.sum(axis=1) == 0].argmax()
#     else:
#         idx = np.array(ks).argmax()
#
#     return (cs[idx],ks[idx])


"""
确定超参数c：0.00019307
重新训练一份，保存模型
"""
import sys

# make a copy of original stdout route
stdout_backup = sys.stdout
# define the log file that receives your log info
log_file = open(r'E:\ScoreCard\cs_model\log\cs_m1_pos_alg03_ks_matrix.log', "w")
# redirect print output to log file
sys.stdout = log_file

print "Now all print info will be written to message.log"
# any command line that you will execute
######################################################

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

clf_path_list = [
    r'E:\ScoreCard\cs_model\gendata\cs_m1_pos_clf_201701.pkl',
    r'E:\ScoreCard\cs_model\gendata\cs_m1_pos_clf_201702.pkl',
    r'E:\ScoreCard\cs_model\gendata\cs_m1_pos_clf_201703.pkl',
    r'E:\ScoreCard\cs_model\gendata\cs_m1_pos_clf_201704.pkl',
    r'E:\ScoreCard\cs_model\gendata\cs_m1_pos_clf_201705.pkl',
    r'E:\ScoreCard\cs_model\gendata\cs_m1_pos_clf_201706.pkl',
    r'E:\ScoreCard\cs_model\gendata\cs_m1_pos_clf_201707.pkl',
    r'E:\ScoreCard\cs_model\gendata\cs_m1_pos_clf_201708.pkl',
    r'E:\ScoreCard\cs_model\gendata\cs_m1_pos_clf_201709.pkl'
]

# save classifier model pkl file
for i in range(dataset_path_list.__len__()):
    dataset_path = 'E:\\ScoreCard\\cs_model\\gendata\\' + dataset_path_list[i]
    config_path = 'E:\\Code\\Python_ML_Code\\cs_model\\config\\config_cs_model_20170'+str(i+1)+'.csv'
    dataset_train = pd.read_csv(dataset_path)
    cfg = pd.read_csv(config_path)
    candidate_var_list = cfg[cfg['is_modelfeature'] == 1]['var_name']

    b = [var for var in dataset_train.columns if sum(dataset_train[var].isnull()) == 0]
    candidate_var_list = list(set(candidate_var_list).intersection(set(b)))

    X_train = dataset_train[candidate_var_list]
    y_train = dataset_train['target']

    clf_l1_LR = LogisticRegression(C=0.00019307, penalty='l1', tol=0.01,class_weight='balanced')
    clf_l1_LR.fit(X_train, y_train)
    # clf_l1_LR.coef_.ravel().copy()

    output = open(clf_path_list[i], 'wb')
    pickle.dump({'features_name':candidate_var_list,'classifier':clf_l1_LR},output)
    output.close()


# prediction
clf_path_list = [
    r'E:\ScoreCard\cs_model\gendata\cs_m1_pos_clf_201701.pkl',
    r'E:\ScoreCard\cs_model\gendata\cs_m1_pos_clf_201702.pkl',
    r'E:\ScoreCard\cs_model\gendata\cs_m1_pos_clf_201703.pkl',
    r'E:\ScoreCard\cs_model\gendata\cs_m1_pos_clf_201704.pkl',
    r'E:\ScoreCard\cs_model\gendata\cs_m1_pos_clf_201705.pkl',
    r'E:\ScoreCard\cs_model\gendata\cs_m1_pos_clf_201706.pkl',
    r'E:\ScoreCard\cs_model\gendata\cs_m1_pos_clf_201707.pkl',
    r'E:\ScoreCard\cs_model\gendata\cs_m1_pos_clf_201708.pkl',
    r'E:\ScoreCard\cs_model\gendata\cs_m1_pos_clf_201709.pkl'
]

ks_matrix = []
for i in range(clf_path_list.__len__()):
    print 'running: ',clf_path_list[i]
    output = open(clf_path_list[i], 'rb')
    clf = pickle.load(output)
    output.close()

    clf_ks = []
    # j 的range范围和i是一样的
    for j in range(clf_path_list.__len__()):
        dataset_path = 'E:\\ScoreCard\\cs_model\\gendata\\' + 'cs_m1_pos_woe_transed_rule_20170'+str(i+1) \
                       +'_features_20170'+str(j+1)+'.csv'
        print dataset_path
        dataset_train = pd.read_csv(dataset_path)

        X_train = dataset_train
        X_train = X_train[clf['features_name']]
        y_train = dataset_train['target']
        print X_train.describe()

        print 'Checking features dtypes:'
        for var in clf['features_name']:
            # fill null
            X_train.loc[X_train[var].isnull(), (var)] = 0

        proba = clf['classifier'].predict_proba(X_train)[:,1]
        ks = compute_ks(proba,y_train)
        print ks
        clf_ks.append(ks)
    print 'ks summary: ',clf_path_list[i],'\n',clf_ks
    ks_matrix.append(clf_ks)
print 'ks_matrix:\n',ks_matrix

######################################################
log_file.close()
# restore the output to initial pattern
sys.stdout = stdout_backup

print "Now this will be presented on screen"
