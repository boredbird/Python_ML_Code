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

    for var in line[5].split(';'):
        wifi_to_shops[var.split('|')[0]][line[1]] += 1

user_shop_behavior.apply(get_wifi_to_shop,axis=1)

from collections import Iterable
isinstance(wifi_to_shops,Iterable) # 判断某个对象是否可迭代

for var in wifi_to_shops.keys():
    for x in wifi_to_shops[var].items():
        if x.__len__()<2:
            print var
            break

wifi_to_shops_nearest = [sorted(wifi_to_shops[var].items(),key=lambda x:int(x[1]),reverse=True)[0][0] for var in wifi_to_shops.keys()]
# wifi_to_shops_nearest.__len__()
# Out[86]:
# 399679
# wifi_to_shops.keys().__len__()
# Out[87]:
# 399679
wifi_to_shops_nearest = pd.Series(wifi_to_shops_nearest,index=wifi_to_shops.keys())  # 每个wifi对应唯一的shop_id
wifi_to_shops_nearest_dict = wifi_to_shops_nearest.to_dict()

# wifi_to_shops_nearest_dict = defaultdict(lambda :'s_666')
# wifi_to_shops_nearest_dict[tuple(wifi_to_shops_nearest.index)] = wifi_to_shops_nearest.values


# 线下
nearest = []
print time.asctime(time.localtime(time.time()))
counter = 0
for line in user_shop_behavior.values:
    counter = counter+1
    if counter%1000 == 0:
        print counter
        print time.asctime(time.localtime(time.time()))

    nearest.append(Counter([wifi_to_shops_nearest[var.split('|')[0]] for var in line[5].split(';')]).most_common(1)[0][0])

print 'ACC: ',sum(user_shop_behavior['shop_id'] == nearest)*1.0/nearest.__len__()
print time.asctime(time.localtime(time.time()))
# ACC:0.698909944069

# 线上预测
nearest = []
print time.asctime(time.localtime(time.time()))
for line in evalset.values:
    nearest.append(
        Counter([wifi_to_shops_nearest_dict.setdefault(var.split('|')[0],'s_666') for var in line[6].split(';')]).most_common(1)[0][0])

print time.asctime(time.localtime(time.time()))

result = pd.DataFrame({'row_id':evalset['row_id'],'shop_id':nearest})
result.to_csv(r'E:\output\submit\BDCI_first_round_alg07_submit01.csv.csv',index=None)

# score:0.6568