# -*- coding:utf-8 -*-
__author__ = 'maomaochong'
"""
对比online learning 和batch learning的效果
"""
import pandas as pd
import time
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from woe.eval import  compute_ks

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
trainset = zip(X_train_enc,y_train)
d = X_train_enc.shape[1]
ks_avg_list = []

clf_l1_LR = LogisticRegression(C=0.01, penalty='l1', tol=0.01,class_weight='balanced')
print '[START]',time.asctime(time.localtime(time.time()))
clf_l1_LR.fit(X_train, y_train)
print '[END]',time.asctime(time.localtime(time.time()))

y_hat = clf_l1_LR.predict_proba(X_train)[:,1]
ks = compute_ks(y_hat,y_train)
print ks
# 0.258477461319

clf_l1_LR = LogisticRegression(C=0.01, penalty='l1', tol=0.01,class_weight='balanced')
print '[START]',time.asctime(time.localtime(time.time()))
clf_l1_LR.fit(X_train_enc, y_train)
print '[END]',time.asctime(time.localtime(time.time()))

y_hat = clf_l1_LR.predict_proba(X_train_enc)[:,1]
ks = compute_ks(y_hat,y_train)
print ks
# 0.287423006797

clf_l1_LR = LogisticRegression(C=0.001, penalty='l1', tol=0.01,class_weight='balanced')
print '[START]',time.asctime(time.localtime(time.time()))
clf_l1_LR.fit(X_train_enc, y_train)
print '[END]',time.asctime(time.localtime(time.time()))

y_hat = clf_l1_LR.predict_proba(X_train_enc)[:,1]
ks = compute_ks(y_hat,y_train)
print ks
# 0.267392474814

# 至此可以确定不是超参的问题，也不是ftrl模型代码的问题
"""
调用ftrl其他人的包，对比下效率和效果
"""




