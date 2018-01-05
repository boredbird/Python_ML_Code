# -*- coding:utf-8 -*-
__author__ = 'maomaochong'
"""
以01变量输入训练模型
"""
from sklearn.preprocessing import OneHotEncoder
import numpy as np
import pandas as pd
import woe.config as config
import woe.feature_process as fp
import pickle
import time
import copy

"""
测试sklearn中的onehotencoder
"""
# enc = OneHotEncoder()
# # X needs to contain only non-negative integers
# enc.fit([[0, 0, 3], [1, 1, 0], [0, 2, 1], [1, 0, 2]])

"""
enc.fit([[0, 0.2, 3], [1, 1, 0], [0, 2, 1], [1, 0, 2]])
Out[18]:
OneHotEncoder(categorical_features='all', dtype=<type 'numpy.float64'>,
       handle_unknown='error', n_values='auto', sparse=True)
print enc.n_values_
print enc.feature_indices_
[2 3 4]
[0 2 5 9]
"""
# print enc.n_values_
# print enc.feature_indices_
# enc.transform([[0, 1, 1]]).toarray()

"""
重新进行woe替换，不是用woe值而是正整数
"""

"""
特征转换为01输入
"""
import pandas as pd
import time
import pickle
import os
from woe.eval import  compute_ks
from woe.ftrl import *
from sklearn.svm import l1_min_c
from multiprocessing import Pool

if __name__ == '__main__':
    for i in range(1,23):
        # i = 0
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
        trainset = zip(X_train_enc,y_train)
        d = X_train_enc.shape[1]
        ks_avg_list = []

        # 初始化
        # ftrl = FTRL(dim=d, l1=0.01, l2=1.0, alpha=0.1, beta=1.0)

        # load
        pkl_path = 'E:\\ScoreCard\\cs_model\\cs_m1_pos_model_daily\\gendata\\LogisticRegression_Model\\'
        recent_path = max(os.listdir(pkl_path))
        recent_path = pkl_path + recent_path
        output = open(recent_path, 'rb')
        ftrl = pickle.load(output)
        output.close()

        # batch train ftrl model
        print('%s\tFTRL MODEL TRAINING START:' % (time.asctime(time.localtime(time.time()))))

        for j in range(50):
            datasubset = trainset[j*10000:(j+1)*10000]
            ftrl.train(datasubset, verbos=0, max_itr=10000, eta=0.01, epochs=100)
            y_hat = 1.0 / (1.0 + np.exp(X_train_enc.dot(ftrl.w)))
            ks = compute_ks(y_hat,y_train)
            print('%s\titer=%s\tks:%s' % (time.asctime(time.localtime(time.time())),str((j+1)*10000),str(ks)))

        # dump
        current_path = pkl_path + 'ftrl_' + time.strftime("%Y%m%d%H%M%S", time.localtime()) +'.pkl'
        print('%s\tDUMP MODEL FILE:\n%s' % (time.asctime(time.localtime(time.time())),current_path))
        output = open(current_path, 'wb')
        pickle.dump(ftrl,output)
        output.close()


