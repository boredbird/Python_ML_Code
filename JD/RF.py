# -*- coding:utf-8 -*-
__author__ = 'maomaochong'
import sys
sys.path.append('E:/Code/Python_ML_Code/JD')
import pandas as pd
import load_data as ld
import time
import numpy as np
from config_params import *
import sklearn
from sklearn.ensemble import  RandomForestClassifier
'''
#拆分训练集和测试集
feature_train, feature_test, target_train, target_test = train_test_split(feature, target, test_size=0.1, random_state=42)

#分类型决策树
clf = RandomForestClassifier(n_estimators=10)
#训练模型
clf = clf.fit(X, Y)
'''
##加载数据集
train_data_file = ['trainset4_kuanbiao','trainset3_kuanbiao','trainset2_kuanbiao','trainset1_kuanbiao']
teset_data_file = ['testset_kuanbiao']
predict_data_file = ['predictset_kuanbiao']


train_dataset = pd.read_csv(gen_data_path + train_data_file[0] + '.csv')
pd.set_option('max_info_columns', 500)
train_dataset.dtypes[train_dataset.dtypes == 'object']

columns = ['acc_rg_day_diff'
,'acc_browse_order_rate'
,'acc_shopcar_order_rate'
,'acc_follow_order_rate'
,'acc_click_order_rate'
,'acc_shopcar_order_rate_gbbrand'
,'acc_follow_order_rate_gbbrand'
,'acc_click_order_rate_gbbrand'
,'acc_sku_cnt'
,'acc_comment_num_avg'
,'acc_has_bad_comment_avg'
,'acc_bad_comment_rate_avg'
,'acc_comment_num_sum'
,'acc_has_bad_comment_sum'
,'acc_bad_comment_rate_sum'
,'acc_has_bad_comment_sku_rate'
,'acc_comment_num'
,'acc_has_bad_comment'
,'acc_bad_comment_rate'
,'acc_browse_order_rate_gbsku'
,'acc_shopcar_order_rate_gbsku'
,'acc_follow_order_rate_gbsku'
,'acc_click_order_rate_gbsku']

for column in columns:
    train_dataset[column] = train_dataset[column].astype('float32')
    print column


clf = RandomForestClassifier(n_estimators=500)
clf = clf.fit(train_X,train_Y)
