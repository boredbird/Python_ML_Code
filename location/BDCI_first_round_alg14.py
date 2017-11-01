#-*- coding:utf-8 -*-

"""
wifi 是人
shops 是物品
信号强度是打分
"""

import pandas as pd
from collections import defaultdict

user_shop_behavior = pd.read_csv(r'E:\output\rawdata\ccf_first_round_user_shop_behavior.csv')
evalution = pd.read_csv(r'E:\output\rawdata\evaluation_public.csv')

#构造规则
wifi_to_shops = defaultdict(lambda : defaultdict(lambda :[]))
for line in user_shop_behavior.values:
    for wifi in line[5].split(';'):
        wifi_to_shops[wifi.split('|')[0]][line[1]].append(int(wifi.split('|')[1]))

wifi_to_shops_cnt = defaultdict(lambda : defaultdict(lambda :0))
for wifi in wifi_to_shops.keys():
    for shop in wifi_to_shops[wifi]:
        wifi_to_shops_cnt[wifi][shop] += wifi_to_shops[wifi][shop].__len__()

wifi_to_shops_rate = defaultdict(lambda : defaultdict(lambda :0))
for wifi in wifi_to_shops_cnt.keys():
    wifi_total_cnt = sum(wifi_to_shops_cnt[wifi].values())
    for shop in wifi_to_shops_cnt[wifi]:
        wifi_to_shops_rate[wifi][shop] = wifi_to_shops_cnt[wifi][shop]*1.0/wifi_total_cnt

#
# wifi = 'b_8512877'
# wifi_to_shops_rate[wifi]

# 线下验证
# 较之前次数相加，改为概率值相加
right_count = 0
for line in user_shop_behavior.values:
    counter = defaultdict(lambda : 0)
    for wifi in line[5].split(';'):
        for k, v in wifi_to_shops_rate[wifi.split('|')[0]].items():
            counter[k] += v

    pred_one = sorted(counter.items(),key=lambda x:x[1],reverse=True)[0][0]
    if pred_one == line[1]:
        right_count += 1
print('acc:',right_count*1.0/len(user_shop_behavior))


# 计算相似度
for line in user_shop_behavior.values:
    for wifi in line[5].split(';'):
        wifi_to_shops[wifi.split('|')[0]][line[1]].append(int(wifi.split('|')[1]))