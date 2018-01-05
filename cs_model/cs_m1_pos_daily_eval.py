# -*- coding:utf-8 -*-
__author__ = 'maomaochong'
import numpy as np
import pandas as pd
import time
import pickle
import os
from woe.eval import  compute_ks
from woe.ftrl import *

class LR(object):
    @staticmethod
    def fn(w, x):
        '''决策函数为sigmoid函数
        '''
        return 1.0 / (1.0 + np.exp(-w.dot(x)))

    @staticmethod
    def loss(y, y_hat):
        '''交叉熵损失函数
        '''
        return np.sum(np.nan_to_num(-y * np.log(y_hat) - (1 - y) * np.log(1 - y_hat)))

    @staticmethod
    def grad(y, y_hat, x):
        '''交叉熵损失函数对权重w的一阶导数
        '''
        return (y_hat - y) * x


class FTRL(object):
    def __init__(self, dim, l1, l2, alpha, beta, decisionFunc=LR):
        self.dim = dim
        self.decisionFunc = decisionFunc
        self.z = np.zeros(dim)
        self.n = np.zeros(dim)
        self.w = np.zeros(dim)
        self.w_list = []
        self.loss_list = []
        self.l1 = l1
        self.l2 = l2
        self.alpha = alpha
        self.beta = beta

    def predict(self, x):
        return self.decisionFunc.fn(self.w, x)

    def update(self, x, y):
        self.w = np.array([0 if np.abs(self.z[i]) <= self.l1 else (np.sign(
            self.z[i]) * self.l1 - self.z[i]) / (self.l2 + (self.beta + np.sqrt(self.n[i])) / self.alpha) for i in xrange(self.dim)])
        y_hat = self.predict(x)
        g = self.decisionFunc.grad(y, y_hat, x)
        sigma = (np.sqrt(self.n + g * g) - np.sqrt(self.n)) / self.alpha
        self.z += g - sigma * self.w
        self.n += g * g
        return self.decisionFunc.loss(y, y_hat)

    def train(self, trainSet, verbos=False, max_itr=10000000000, eta=0.01, epochs=100):
        itr = 0
        n = 0
        while True:
            for x, y in trainSet:
                loss = self.update(x, y)
                if verbos and n%verbos==0:
                    print "itr=" + str(n) + "\tloss=" + str(loss)
                    self.w_list.append(self.w)
                    self.loss_list.append(loss)
                if loss < eta:
                    itr += 1
                else:
                    itr = 0
                if itr >= epochs:  # 损失函数已连续epochs次迭代小于eta
                    print "loss have less than", eta, " continuously for ", itr, "iterations"
                    return
                n += 1
                if n >= max_itr:
                    print "reach max iteration", max_itr
                    return

pkl_path = 'E:\\ScoreCard\\cs_model\\cs_m1_pos_model_daily\\gendata\\LogisticRegression_Model\\'
recent_path_list = os.listdir(pkl_path)
i = 0
recent_path = pkl_path + recent_path_list[i]
output = open(recent_path, 'rb')
ftrl = pickle.load(output)
output.close()

validation_dataset_path = 'E:\\ScoreCard\\cs_model\\cs_m1_pos_model_daily\\gendata\\dataset_split_by_rows\\m1_rsx_cs_unify_model_features_201705_daily_24_woe_transed.csv'
validation_dataset = pd.read_csv(validation_dataset_path)

config_path = 'E:\\Code\\Python_ML_Code\\cs_model\\config\\config_cs_daily_model.csv'
print('%s\tLOAD CONFIG FILE:\n%s' % (time.asctime(time.localtime(time.time())),config_path))
cfg = pd.read_csv(config_path)
candidate_var_list = cfg[cfg['is_modelfeature'] == 1]['var_name']

b = [var for var in validation_dataset.columns if sum(validation_dataset[var].isnull()) == 0]
candidate_var_list = list(set(candidate_var_list).intersection(set(b)))

X_train = validation_dataset[candidate_var_list].values
y_train = validation_dataset['target'].values
# y_hat = ftrl.predict(X_train)
y_hat = 1.0 / (1.0 + np.exp(X_train.dot(ftrl.w)))
ks = compute_ks(y_hat,y_train)
print ks

# y_hat = clf_l1_LR.predict_proba(X_train)[:,1]
# loss = LR.loss(y_train,y_hat)/y_hat.__len__()
# print loss


