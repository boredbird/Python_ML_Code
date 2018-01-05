# -*- coding:utf-8 -*-
__author__ = 'maomaochong'
"""
warm up
"""
from sklearn.preprocessing import OneHotEncoder
import pandas as pd
import time
from woe.eval import  compute_ks
from woe.ftrl import *
from sklearn.svm import l1_min_c
from multiprocessing import Pool

# def warm_up(trainset,d,l1,alpha):
def warm_up(params_list):
    X_train,y_train,d,l1,alpha = params_list
    trainset = zip(X_train,y_train)
    # 初始化
    ks_list = []
    ftrl = FTRL(dim=d, l1=l1, l2=1.0, alpha=alpha, beta=1.0)
    print('%s\tFTRL MODEL TRAINING ROUND:[l1]\t%s\t[alpha]\t%s\t' % (time.asctime(time.localtime(time.time())),str(l1),str(alpha)))
    for j in range(50):# 50
        datasubset = trainset[j*10000:(j+1)*10000]
        ftrl.train(datasubset, verbos=0, max_itr=10000, eta=0.01, epochs=100)
        y_hat = 1.0 / (1.0 + np.exp(X_train.dot(ftrl.w)))
        ks = compute_ks(y_hat,y_train)
        ks_list.append(ks)
        print('%s\t[l1]\t%s\t[alpha]\t%s\titer=%s\tks:%s' % (time.asctime(time.localtime(time.time())),str(l1),str(alpha),str((j+1)*10000),str(ks)))

    print('%s\tFTRL MODEL TRAINING ROUND:[l1]\t%s\t[alpha]\t%s\t[ks_avg]\t%s' % (time.asctime(time.localtime(time.time())),str(l1),str(alpha),str(sum(ks_list)/ks_list.__len__())))
    return sum(ks_list)/ks_list.__len__()

if __name__ == '__main__':
    # for i in range(1,23):
    i = 0
    dataset_path = 'E:\\ScoreCard\\cs_model\\cs_m1_pos_model_daily\\gendata\\dataset_split_by_rows\\' \
                   + 'm1_rsx_cs_unify_model_features_201705_daily_' \
                   + str(i+1) + '_woe01_transed.csv'

    print('%s\tLOAD DATASET FILE:\n%s' % (time.asctime(time.localtime(time.time())),dataset_path))
    dataset = pd.read_csv(dataset_path)

    config_path = 'E:\\Code\\Python_ML_Code\\cs_model\\config\\config_cs_daily_model.csv'
    print('%s\tLOAD CONFIG FILE:\n%s' % (time.asctime(time.localtime(time.time())),config_path))
    cfg = pd.read_csv(config_path)
    candidate_var_list = cfg[cfg['is_modelfeature'] == 1]['var_name']

    b = [var for var in dataset.columns if sum(dataset[var].isnull()) == 0]
    candidate_var_list = list(set(candidate_var_list).intersection(set(b)))

    X_train = dataset[candidate_var_list].values
    y_train = dataset['target'].values

    enc = OneHotEncoder()
    enc.fit(X_train)
    print 'enc.n_values_:\n',enc.n_values_
    print 'enc.feature_indices_:\n',enc.feature_indices_

    X_train_enc = enc.transform(X_train).toarray()

    d = X_train_enc.shape[1]
    alpha_list = np.logspace(0, -5,6)
    l1_list = l1_min_c(X_train_enc, y_train, loss='log') * np.logspace(0, 9,10)
    print 'alpha_list:\n',alpha_list
    print 'l1_list:\n',l1_list
    ks_avg_list = []
    params_list = []
    for l1 in l1_list:
        for alpha in alpha_list:
            params_list.append((X_train_enc,y_train,d,l1,alpha))

    pool = Pool(processes=4)
    ks_avg_list = pool.map(warm_up,params_list)
    pool.close()
    pool.join()

    idx = 0
    for l1 in l1_list:
        for alpha in alpha_list:
            print('[l1]\t%s\t[alpha]\t%s\t[ks_avg]\t%s' % (str(l1),str(alpha),str(ks_avg_list[idx])))
            idx += 1

"""
ftrl_training_warmup_01.log:
    alpha_list = np.logspace(0, -5,20)
    l1_list = l1_min_c(X_train_enc, y_train, loss='log') * np.logspace(0, 9,100)
ftrl_training_warmup_02.log:
    alpha_list = np.logspace(0, -5,6)
    l1_list = l1_min_c(X_train_enc, y_train, loss='log') * np.logspace(0, 9,10)

l1:0.01
alpha:0.1
"""


