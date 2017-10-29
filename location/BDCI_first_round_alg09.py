# -*- coding:utf-8 -*-
"""
根据ccf_first_round_user_shop_behavior用户行为发生时的每个wifi所对应的shop_id,
按照WiFi分组取出现次数个数最多的shop_id,作为wifi_id与shop_id的对应关系；此时一个wifi_id取信号最强的前k个shop_id;不统计次数
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
from collections import Counter

user_shop_behavior = pd.read_csv(r'E:\output\rawdata\ccf_first_round_user_shop_behavior.csv')
shop_info = pd.read_csv(r'E:\output\rawdata\ccf_first_round_shop_info.csv')
evalset = pd.read_csv(r'E:\output\rawdata\evaluation_public.csv')

nearest = []
shop_info.index = shop_info['shop_id']

user_shop_behavior['mall_id'] = shop_info.loc[user_shop_behavior['shop_id'] ,]['mall_id'].tolist()
user_shop_behavior.index = user_shop_behavior['mall_id']

# 训练好的规则
wifi_to_shops = defaultdict(lambda : defaultdict(lambda :0))  # 每个wifi出现过的所有的shop_id

counter = 0
def get_wifi_to_shop(line):
    global counter
    counter = counter+1
    if counter%10000 == 0:
        print counter
        print time.asctime(time.localtime(time.time()))

    # 此处利用信号强度信息，信号最强的作为有效的统计
    for wifi in sorted([(var.split('|')[0],var.split('|')[1]) for var in line[5].split(';')],key=lambda x:int(x[1]),reverse=True)[:3]:
        wifi_to_shops[wifi[0]][line[1]] += 1


user_shop_behavior.apply(get_wifi_to_shop,axis=1)

from collections import Iterable
isinstance(wifi_to_shops,Iterable) # 判断某个对象是否可迭代

wifi_to_shops_nearest = [sorted(wifi_to_shops[var].items(),key=lambda x:int(x[1]),reverse=True)[0][0] for var in wifi_to_shops.keys()]
# wifi_to_shops_nearest.__len__()
# Out[86]:
# 171766
# wifi_to_shops.keys().__len__()
# Out[87]:
# 171766

evalset_wifi_to_shops = defaultdict(lambda : defaultdict(lambda :0))  # 每个wifi出现过的所有的shop_id

counter = 0
def evalset_get_wifi_to_shop(line):
    global counter
    counter = counter+1
    if counter%10000 == 0:
        print counter
        print time.asctime(time.localtime(time.time()))

    # 此处利用信号强度信息，信号最强的作为有效的统计
    for wifi in sorted([(var.split('|')[0],var.split('|')[1]) for var in line[6].split(';')],key=lambda x:int(x[1]),reverse=True)[:3]:
        evalset_wifi_to_shops[wifi[0]][line[1]] += 1

evalset.apply(evalset_get_wifi_to_shop,axis=1)
print evalset_wifi_to_shops.keys().__len__()
# 101046

# 交集
ret_list = list((set(wifi_to_shops_nearest).union(set(evalset_wifi_to_shops)))^(set(wifi_to_shops_nearest)^set(evalset_wifi_to_shops)))
ret_list.__len__()
# 0

