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


dataset = pd.read_csv('G:/gen_data/testset_kuanbiao.csv')

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
    # train_dataset[column] = train_dataset[column].astype('float32')
    train_dataset[column].convert_objects(convert_numeric=True).dtypes
    print column

train_dataset['acc_rg_day_diff'] = train_dataset['acc_rg_day_diff'].astype('float32')


train_Y = train_dataset['mark']
train_X = train_dataset[['user_id'
,'sku_id'
,'cate'
,'brand'
,'browse_cnt'
,'shoppingcar_in_cnt'
,'shoppingcar_out_cnt'
,'order_cnt'
,'follow_cnt'
,'click_cnt'
,'browse_cnt_gbuser'
,'shoppingcar_in_cnt_gbuser'
,'shoppingcar_out_cnt_gbuser'
,'order_cnt_gbuser'
,'follow_cnt_gbuser'
,'click_cnt_gbuser'
,'browse_cnt_gbsku'
,'shoppingcar_in_cnt_gbsku'
,'shoppingcar_out_cnt_gbsku'
,'order_cnt_gbsku'
,'follow_cnt_gbsku'
,'click_cnt_gbsku'
,'browse_cnt_gbcate'
,'shoppingcar_in_cnt_gbcate'
,'shoppingcar_out_cnt_gbcate'
,'order_cnt_gbcate'
,'follow_cnt_gbcate'
,'click_cnt_gbcate'
,'acc_user_id'
,'acc_age'
,'acc_sex'
,'acc_user_lv_cd'
,'acc_rg_day_diff'
,'acc_browse_cnt'
,'acc_shoppingcar_in_cnt'
,'acc_shoppingcar_out_cnt'
,'acc_order_cnt'
,'acc_follow_cnt'
,'acc_browse_order_rate'
,'acc_shopcar_order_rate'
,'acc_follow_order_rate'
,'acc_click_order_rate'
,'acc_brand'
,'acc_browse_cnt_gbbrand'
,'acc_shoppingcar_in_cnt_gbbrand'
,'acc_shoppingcar_out_cnt_gbbrand'
,'acc_order_cnt_gbbrand'
,'acc_follow_cnt_gbbrand'
,'acc_browse_order_rate_gbbrand'
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
,'acc_cate'
,'acc_browse_cnt_gbcate'
,'acc_shoppingcar_in_cnt_gbcate'
,'acc_shoppingcar_out_cnt_gbcate'
,'acc_order_cnt_gbcate'
,'acc_follow_cnt_gbcate'
,'acc_browse_order_rate_gbcate'
,'acc_shopcar_order_rate_gbcate'
,'acc_follow_order_rate_gbcate'
,'acc_click_order_rate_gbcate'
,'acc_sku_id'
,'acc_attr1'
,'acc_attr2'
,'acc_attr3'
,'acc_cate_gbsku'
,'acc_brand_gbsku'
,'acc_comment_num'
,'acc_has_bad_comment'
,'acc_bad_comment_rate'
,'acc_browse_cnt_gbsku'
,'acc_shoppingcar_in_cnt_gbsku'
,'acc_shoppingcar_out_cnt_gbsku'
,'acc_order_cnt_gbsku'
,'acc_follow_cnt_gbsku'
,'acc_browse_order_rate_gbsku'
,'acc_shopcar_order_rate_gbsku'
,'acc_follow_order_rate_gbsku'
,'acc_click_order_rate_gbsku'
,'acc_brand_gbbs'
,'acc_sku_id_gbbs'
,'acc_browse_cnt_sum'
,'acc_shoppingcar_in_cnt_sum'
,'acc_shoppingcar_out_cnt_sum'
,'acc_order_cnt_sum'
,'acc_follow_cnt_sum'
,'acc_click_cnt_sum'
,'acc_browse_cnt_sum_rank'
,'acc_click_cnt_sum_rank'
,'acc_follow_cnt_sum_rank'
,'acc_order_cnt_sum_rank'
,'acc_shoppingcar_in_cnt_sum_rank'
,'acc_shoppingcar_out_cnt_sum_rank'
,'acc_cate_gbcb'
,'acc_brand_gbcb'
,'acc_browse_cnt_sum_gbcb'
,'acc_shoppingcar_in_cnt_sum_gbcb'
,'acc_shoppingcar_out_cnt_sum_gbcb'
,'acc_order_cnt_sum_gbcb'
,'acc_follow_cnt_sum_gbcb'
,'acc_click_cnt_sum_gbcb'
,'acc_browse_cnt_sum_rank_gbcb'
,'acc_click_cnt_sum_rank_gbcb'
,'acc_follow_cnt_sum_rank_gbcb'
,'acc_order_cnt_sum_rank_gbcb'
,'acc_shoppingcar_in_cnt_sum_rank_gbcb'
,'acc_shoppingcar_out_cnt_sum_rank_gbcb'
,'acc_cate_gbcs'
,'acc_sku_id_gbcs'
,'acc_browse_cnt_sum_gbcs'
,'acc_shoppingcar_in_cnt_sum_gbcs'
,'acc_shoppingcar_out_cnt_sum_gbcs'
,'acc_order_cnt_sum_gbcs'
,'acc_follow_cnt_sum_gbcs'
,'acc_click_cnt_sum_gbcs'
,'acc_browse_cnt_sum_rank_gbcs'
,'acc_click_cnt_sum_rank_gbcs'
,'acc_follow_cnt_sum_rank_gbcs'
,'acc_order_cnt_sum_rank_gbcs'
,'acc_shoppingcar_in_cnt_sum_rank_gbcs'
,'acc_shoppingcar_out_cnt_sum_rank_gbcs'
,'if_order_cnt']]

train_X = train_X.drop([columns], axis=1)
clf = RandomForestClassifier(n_estimators=500)
clf = clf.fit(train_X,train_Y)