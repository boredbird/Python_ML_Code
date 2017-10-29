# -*- coding:utf-8 -*-
"""
根据ccf_first_round_user_shop_behavior用户行为发生时的每个wifi所对应的shop_id,
按照WiFi分组取出现次数个数最多的shop_id,作为wifi_id与shop_id的对应关系；此时一个wifi_id出现过的shop_id都在里面;相比alg06去掉次数信息
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

# 线下
nearest = []
print time.asctime(time.localtime(time.time()))
counter = 0
for line in user_shop_behavior.values:
    counter = counter+1
    if counter%1000 == 0:
        print counter
        print time.asctime(time.localtime(time.time()))

    a = []
    for var in line[5].split(';'):
        a.extend(wifi_to_shops[var.split('|')[0]].items())

    nearest.append(np.argmax(pd.DataFrame(a).groupby(0).count()[1])) # 较alg06,此处sum()改为count()

print 'ACC: ',sum(user_shop_behavior['shop_id'] == nearest)*1.0/nearest.__len__()
print time.asctime(time.localtime(time.time()))
# ACC:  0.689132392807

# 线上预测
nearest = []
print time.asctime(time.localtime(time.time()))
for line in evalset.values:
    a = []
    for var in line[6].split(';'):
        a.extend(wifi_to_shops[var.split('|')[0]].items())

    if a == []:
        a = [('s_666',0),]

    nearest.append(np.argmax(pd.DataFrame(a).groupby(0).count()[1])) # 较alg06,此处sum()改为count()

print time.asctime(time.localtime(time.time()))

result = pd.DataFrame({'row_id':evalset['row_id'],'shop_id':nearest})
result.to_csv(r'E:\output\submit\BDCI_first_round_alg08_submit01.csv.csv',index=None)

# score:0.5555