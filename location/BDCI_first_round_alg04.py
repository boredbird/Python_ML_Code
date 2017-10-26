# -*- coding:utf-8 -*-
"""
1、根据ccf_first_round_user_shop_behavior用户行为发生时的经纬度所对应的shop_id
evaluation_public数据集匹配user_shop_behavior经纬度，找到最近的K个shop_id，
确定参数k，构造负样本:
通过参数k来确定负样本是没有意义的，user_shop_behavior用户行为发生时的shop_id就是真实的shop_id，只取前k个距离最近的shop_id肯定会造成召回率很低
2、取user_shop_behavior用户行为发生时的经纬度平均值，确定每个shop_id的经纬度坐标；
然后evaluation_public数据集匹配这个每个shop_id的经纬度坐标（相当于shop_info那张表）,最近的shop_id作为结果提交。
"""
__author__ = 'maomaochong'
import pandas as pd
from location.util import  *
import time
from multiprocessing import Pool
from collections import Counter
import numpy as np

user_shop_behavior = pd.read_csv(r'E:\output\rawdata\ccf_first_round_user_shop_behavior.csv')
shop_info = pd.read_csv(r'E:\output\rawdata\ccf_first_round_shop_info.csv')
evalset = pd.read_csv(r'E:\output\rawdata\evaluation_public.csv')

nearest = []
shop_info.index = shop_info['shop_id']

user_shop_behavior['mall_id'] = shop_info.loc[user_shop_behavior['shop_id'] ,]['mall_id'].tolist()
user_shop_behavior.index = user_shop_behavior['mall_id']

counter = 0
shop_xy = user_shop_behavior[['longitude', 'latitude', 'shop_id','mall_id']].groupby(['mall_id','shop_id']).mean()

def get_nearest(line):
    # print line
    # print type(line)
    global counter
    global shop_xy
    counter = counter+1
    if counter%1000 == 0:
        print counter
        print time.asctime(time.localtime(time.time()))

    shop_distance = [[get_distance_hav(line[3],line[4],shop[0][0],shop[0][1]),shop[1]] for shop in zip(shop_xy.loc[line[6],].values,shop_xy.loc[line[6],].index)]
    # print shop_distance
    nearest.append(sorted(shop_distance)[0][1])
    # print sorted(shop_distance)
    # print sorted(shop_distance)[0][1]

# 线下验证
a = user_shop_behavior.apply(get_nearest,axis=1)
print 'ACC: ',sum(user_shop_behavior['shop_id'] == nearest)*1.0/nearest.__len__()
# ACC: 0.311832445091

result = pd.DataFrame({'behavior_mall_id':user_shop_behavior['mall_id']
                          ,'behavior_shop_id':user_shop_behavior['shop_id']
                          ,'nearest_shop_id':nearest})
result.to_csv(r'E:\output\gendata\BDCI_first_round_alg04_behavior_shop_avgxy.csv.csv',index=None)

result['same'] = result['behavior_shop_id']==result['nearest_shop_id']
result_mall_agg = pd.DataFrame()
result_mall_agg['same_cnt'] = result.groupby(['behavior_mall_id'])['same'].sum()
result_mall_agg['total_cnt'] = result.groupby(['behavior_mall_id'])['same'].count()
result_mall_agg['same_rate'] = result_mall_agg['same_cnt']*1.0/result_mall_agg['total_cnt']
result_mall_agg.to_csv(r'E:\output\gendata\BDCI_first_round_alg04_behavior_mall_agg.csv.csv',index=None)

# 效果比较差，未做线上测试。