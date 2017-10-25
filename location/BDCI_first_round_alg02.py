# -*- coding:utf-8 -*-
"""
根据ccf_first_round_user_shop_behavior用户行为发生时的经纬度所对应的shop_id
evaluation_public数据集匹配user_shop_behavior经纬度，找到最近的shop_id作为结果提交
"""
__author__ = 'maomaochong'
import pandas as pd
from util import  *

user_shop_behavior = pd.read_csv(r'E:\output\rawdata\ccf_first_round_user_shop_behavior.csv')
shop_info = pd.read_csv(r'E:\output\rawdata\ccf_first_round_shop_info.csv')
evalset = pd.read_csv(r'E:\output\rawdata\evaluation_public.csv')

nearest = []
for line in evalset.values:
    shop_distance = [[get_distance_hav(line[4],line[5],shop[0],shop[1]),shop[2]] for shop in shop_info[shop_info['mall_id'] == line[2]][['longitude','latitude','shop_id']].values]
    nearest.append(sorted(shop_distance)[0][1])

result = pd.DataFrame({'row_id':evalset.row_id,'shop_id':nearest})
result.to_csv(r'E:\output\submit\BDCI_first_round_alg01_submit01.csv.csv',index=None)


