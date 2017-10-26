# -*- coding:utf-8 -*-
"""
根据ccf_first_round_user_shop_behavior用户行为发生时信号最强的wifi所对应的shop_id,
按照WiFi分组取出现次数个数最多的shop_id,作为wifi_id与shop_id的对应关系；
evaluation_public数据集信号最强的wifi匹配对应的shop_id作为结果提交
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

def get_strongest_wifi(line):
    str = line[5]
    return sorted([var.split('|') for var in str.split(';')], key=lambda x: int(x[1]), reverse=True)[0][0]


line = user_shop_behavior.iloc[0,]
user_shop_behavior['strongest_wifi'] = user_shop_behavior.apply(get_strongest_wifi, axis=1)
a = user_shop_behavior[['strongest_wifi','shop_id']].groupby(['strongest_wifi']).count()
"""
a.shape
Out[52]:
(75879, 1)
"""

def get_strongest_wifi2(line):
    str = line[6]
    return sorted([var.split('|') for var in str.split(';')], key=lambda x: int(x[1]), reverse=True)[0][0]

evalset['strongest_wifi'] = evalset.apply(get_strongest_wifi2, axis=1)
b = np.unique(evalset['strongest_wifi'])
"""
b.shape
Out[58]:
(46764L,)
"""
# 交集
c = np.intersect1d(np.array(a),b)
"""
c.shape
Out[60]:
(0L,)
"""

# 尴尬，完全没有交集。此方案不可行。

# wifi_id 是不是在变啊？？为什么会信号最强的wifi完全不重合。。

