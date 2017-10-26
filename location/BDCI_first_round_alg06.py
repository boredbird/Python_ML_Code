# -*- coding:utf-8 -*-
"""
根据ccf_first_round_user_shop_behavior用户行为发生时的每个wifi所对应的shop_id,
按照WiFi分组取出现次数个数最多的shop_id,作为wifi_id与shop_id的对应关系；此时一个wifi_id对应唯一的shop_id;
evaluation_public数据集中每个wifi匹配对应的shop_id，对应次数最多的shop_id作为结果提交
"""
__author__ = 'maomaochong'
import pandas as pd
from location.util import  *
import time
from multiprocessing import Pool
from collections import Counter
import numpy as np
from collections import defaultdict

user_shop_behavior = pd.read_csv(r'E:\output\rawdata\ccf_first_round_user_shop_behavior.csv')
shop_info = pd.read_csv(r'E:\output\rawdata\ccf_first_round_shop_info.csv')
evalset = pd.read_csv(r'E:\output\rawdata\evaluation_public.csv')

nearest = []
shop_info.index = shop_info['shop_id']

user_shop_behavior['mall_id'] = shop_info.loc[user_shop_behavior['shop_id'] ,]['mall_id'].tolist()
user_shop_behavior.index = user_shop_behavior['mall_id']

wifi_to_shops = defaultdict(lambda : defaultdict(lambda :0))

def get_wifi_to_shop(line):
    for var in line[5].split(';'):
        for wifi in var.split('|'):
            wifi_to_shops[wifi[0]][line[1]] += 1

user_shop_behavior.apply(get_wifi_to_shop,axis=1)