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

"""
全部wifi
"""
#构造规则：
wifi_to_shops = defaultdict(lambda : defaultdict(lambda :[]))
for line in user_shop_behavior.values:
    for wifi in line[5].split(';'):
        wifi_to_shops[wifi.split('|')[0]][line[1]].append(int(wifi.split('|')[1]))

# 全部次数累加
wifi_to_shops_cnt = defaultdict(lambda : defaultdict(lambda :0))
for wifi in wifi_to_shops.keys():
    for shop in wifi_to_shops[wifi]:
        wifi_to_shops_cnt[wifi][shop] += wifi_to_shops[wifi][shop].__len__()

# 全部概率累加
wifi_to_shops_rate = defaultdict(lambda : defaultdict(lambda :0))
for wifi in wifi_to_shops_cnt.keys():
    wifi_total_cnt = sum(wifi_to_shops_cnt[wifi].values())
    for shop in wifi_to_shops_cnt[wifi]:
        wifi_to_shops_rate[wifi][shop] = wifi_to_shops_cnt[wifi][shop]*1.0/wifi_total_cnt

#
# wifi = 'b_8512877'
# wifi_to_shops_rate[wifi]

# 线下验证
# 规则：全部次数累加；预测:取全部wifi
right_count = 0
for line in user_shop_behavior.values:
    counter = defaultdict(lambda : 0)
    for wifi in line[5].split(';'):
        for k, v in wifi_to_shops_cnt[wifi.split('|')[0]].items():
            counter[k] += v

    pred_one = sorted(counter.items(),key=lambda x:x[1],reverse=True)[0][0]
    if pred_one == line[1]:
        right_count += 1
print('acc:',right_count*1.0/len(user_shop_behavior))
# ('acc:', 0.6594306753425921)

# 规则：全部概率累加；预测:取全部wifi
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
# ('acc:', 0.760736018418035)

# 规则：全部概率累加；预测:只取最强信号强度的wifi
# 只取信号最强的，概率值相加
right_count = 0
for line in user_shop_behavior.values:
    wifi = sorted([wifi.split('|') for wifi in line[5].split(';')],key=lambda x:int(x[1]),reverse=True)[0]
    counter = defaultdict(lambda : 0)
    for k,v in wifi_to_shops_rate[wifi[0]].items():
        counter[k] += v
    pred_one = sorted(counter.items(),key=lambda x:x[1],reverse=True)[0][0]
    if pred_one == line[1]:
        right_count += 1
print('acc:',right_count*1.0/len(user_shop_behavior))
# ('acc:', 0.7289367890581407)

# 规则：全部次数累加；预测:只取最强信号强度的wifi
# 只取信号最强的，次数相加
right_count = 0
for line in user_shop_behavior.values:
    wifi = sorted([wifi.split('|') for wifi in line[5].split(';')],key=lambda x:int(x[1]),reverse=True)[0]
    counter = defaultdict(lambda : 0)
    for k,v in wifi_to_shops_cnt[wifi[0]].items():
        counter[k] += v
    pred_one = sorted(counter.items(),key=lambda x:x[1],reverse=True)[0][0]
    if pred_one == line[1]:
        right_count += 1
print('acc:',right_count*1.0/len(user_shop_behavior))
# ('acc:', 0.7290237826390689)


# 构造规则：
# 最强次数累加
strong_wifi_to_shops = defaultdict(lambda : defaultdict(lambda :0))
for line in user_shop_behavior.values:
    wifi = sorted([wifi.split('|') for wifi in line[5].split(';')],key=lambda x:int(x[1]),reverse=True)[0]
    strong_wifi_to_shops[wifi[0]][line[1]] = strong_wifi_to_shops[wifi[0]][line[1]] + 1

# 最强概率累加
strong_wifi_to_shops_rate = defaultdict(lambda : defaultdict(lambda :0))
for wifi in strong_wifi_to_shops.keys():
    wifi_total_cnt = sum(strong_wifi_to_shops[wifi].values())
    for shop in strong_wifi_to_shops[wifi]:
        strong_wifi_to_shops_rate[wifi][shop] = strong_wifi_to_shops[wifi][shop]*1.0/wifi_total_cnt

"""
strong_wifi_to_shops.items().__len__()
Out[86]:
75879

morethanone_cnt = 0
for wifi in strong_wifi_to_shops.keys():
    if strong_wifi_to_shops[wifi].items().__len__() > 1:
        morethanone_cnt += 1

morethanone_cnt
Out[90]:
17981
"""

# 规则：最强次数累加；预测:最强wifi
right_count = 0
for line in user_shop_behavior.values:
    wifi = sorted([wifi.split('|') for wifi in line[5].split(';')],key=lambda x:int(x[1]),reverse=True)[0]
    counter = defaultdict(lambda : 0)
    for k,v in strong_wifi_to_shops[wifi[0]].items():
        counter[k] += v
    pred_one = sorted(counter.items(),key=lambda x:x[1],reverse=True)[0][0]
    if pred_one == line[1]:
        right_count += 1
print('acc:',right_count*1.0/len(user_shop_behavior)) #线下验证
# ('acc:', 0.8704691941670365)

# 规则：最强次数累加；预测:全部wifi
right_count = 0
for line in user_shop_behavior.values:
    counter = defaultdict(lambda : 0)
    for wifi in line[5].split(';'):
        for k, v in strong_wifi_to_shops[wifi.split('|')[0]].items():
            counter[k] += v

    pred_one = sorted(counter.items(),key=lambda x:x[1],reverse=True)[0][0]
    if pred_one == line[1]:
        right_count += 1
print('acc:',right_count*1.0/len(user_shop_behavior))
# ('acc:', 0.6761000514052978)


# 规则：最强概率累加；预测:全部wifi
right_count = 0
for line in user_shop_behavior.values:
    counter = defaultdict(lambda : 0)
    for wifi in line[5].split(';'):
        for k, v in strong_wifi_to_shops_rate[wifi.split('|')[0]].items():
            counter[k] += v

    pred_one = sorted(counter.items(),key=lambda x:x[1],reverse=True)[0][0]
    if pred_one == line[1]:
        right_count += 1
print('acc:',right_count*1.0/len(user_shop_behavior))
# ('acc:', 0.6899223648194444)

# 规则：最强概率累加；预测:最强wifi
right_count = 0
for line in user_shop_behavior.values:
    wifi = sorted([wifi.split('|') for wifi in line[5].split(';')],key=lambda x:int(x[1]),reverse=True)[0]
    counter = defaultdict(lambda : 0)
    for k,v in strong_wifi_to_shops_rate[wifi[0]].items():
        counter[k] += v
    pred_one = sorted(counter.items(),key=lambda x:x[1],reverse=True)[0][0]
    if pred_one == line[1]:
        right_count += 1
print('acc:',right_count*1.0/len(user_shop_behavior)) #线下验证
# ('acc:', 0.8704691941670365)

# 发现：
# 在规则概率值累加的情况下：预测只取最强信号强度的wifi 比 取全部wifi的效果要差
# 在只取信号最强的情况下：规则次数全部累加 与 概率全部累加效果差不多

"""
聚合操作，max
"""
# 规则：最强概率max；预测:最强wifi
right_count = 0
for line in user_shop_behavior.values:
    wifi = sorted([wifi.split('|') for wifi in line[5].split(';')],key=lambda x:int(x[1]),reverse=True)[0]
    max_proba = defaultdict(lambda : 0)
    for k,v in strong_wifi_to_shops_rate[wifi[0]].items():
        max_proba[k] = max(max_proba[k],v)

    pred_one = sorted(max_proba.items(),key=lambda x:x[1],reverse=True)[0][0]
    if pred_one == line[1]:
        right_count += 1
print('acc:',right_count*1.0/len(user_shop_behavior))
# ('acc:', 0.8704691941670365)

# pred_one = sorted(strong_wifi_to_shops_rate[wifi[0]].items(),key=lambda x:int(x[1]),reverse=True)[0][0]
# ('acc:', 0.640136553560366)

# 规则：全部概率max；预测:全部wifi
right_count = 0
for line in user_shop_behavior.values:
    max_proba = defaultdict(lambda : 0)
    for wifi in line[5].split(';'):
        for k,v in wifi_to_shops_rate[wifi.split('|')[0]].items():
            max_proba[k] = max(max_proba[k],v)

    pred_one = sorted(max_proba.items(),key=lambda x:x[1],reverse=True)[0][0]
    if pred_one == line[1]:
        right_count += 1
print('acc:',right_count*1.0/len(user_shop_behavior))
# ('acc:', 0.7471878665922681)

# 规则：全部概率avg；预测:全部wifi
right_count = 0
for line in user_shop_behavior.values:
    proba = defaultdict(lambda : [])
    for wifi in line[5].split(';'):
        for k,v in wifi_to_shops_rate[wifi.split('|')[0]].items():
            proba[k].append(v)

    max_proba = defaultdict(lambda : [])
    for wifi in proba.keys():
        max_proba[wifi] = sum(proba[wifi])*1.0/proba[wifi].__len__()

    pred_one = sorted(max_proba.items(),key=lambda x:x[1],reverse=True)[0][0]
    if pred_one == line[1]:
        right_count += 1
print('acc:',right_count*1.0/len(user_shop_behavior))
# ('acc:', 0.7093482950576223)


# 构造规则：
# 最强强度
strong_wifi_to_shops = defaultdict(lambda : defaultdict(lambda :0))
for line in user_shop_behavior.values:
    wifi = sorted([wifi.split('|') for wifi in line[5].split(';')],key=lambda x:int(x[1]),reverse=True)[0]
    strong_wifi_to_shops[wifi[0]][line[1]] = strong_wifi_to_shops[wifi[0]][line[1]] + 1

# 最强强度
strong_wifi_to_shops_rate = defaultdict(lambda : defaultdict(lambda :0))
for wifi in strong_wifi_to_shops.keys():
    wifi_total_cnt = sum(strong_wifi_to_shops[wifi].values())
    for shop in strong_wifi_to_shops[wifi]:
        strong_wifi_to_shops_rate[wifi][shop] = strong_wifi_to_shops[wifi][shop]*1.0/wifi_total_cnt

# 规则：全部强度max；预测:全部wifi
right_count = 0
for line in user_shop_behavior.values:
    max_proba = defaultdict(lambda : 0)
    for wifi in line[5].split(';'):
        for k,v in wifi_to_shops_rate[wifi.split('|')[0]].items():
            max_proba[k] = max(max_proba[k],v)

    pred_one = sorted(max_proba.items(),key=lambda x:x[1],reverse=True)[0][0]
    if pred_one == line[1]:
        right_count += 1
print('acc:',right_count*1.0/len(user_shop_behavior))
# ('acc:', 0.)














"""
构建规则：以wifi_id在shop_id的信号强度均值
"""
#构造规则：
signal_wifi_to_shops = defaultdict(lambda : defaultdict(lambda :[]))
for line in user_shop_behavior.values:
    for wifi in line[5].split(';'):
        signal_wifi_to_shops[wifi.split('|')[0]][line[1]].append(int(wifi.split('|')[1]))

# 全部信号平均
signal_wifi_to_shops_avg = defaultdict(lambda : defaultdict(lambda :0))
for wifi in signal_wifi_to_shops.keys():
    for shop in signal_wifi_to_shops[wifi]:
        signal_wifi_to_shops_avg[wifi][shop] = sum(signal_wifi_to_shops[wifi][shop])*1.0/signal_wifi_to_shops[wifi][shop].__len__()

# 规则：最强信号累加；预测:最强wifi
right_count = 0
for line in user_shop_behavior.values:
    wifi = sorted([wifi.split('|') for wifi in line[5].split(';')],key=lambda x:int(x[1]),reverse=True)[0]
    counter = defaultdict(lambda : 0)
    for k,v in signal_wifi_to_shops_avg[wifi[0]].items():
        counter[k] += v
    pred_one = sorted(counter.items(),key=lambda x:x[1],reverse=True)[0][0]
    if pred_one == line[1]:
        right_count += 1
print('acc:',right_count*1.0/len(user_shop_behavior)) #线下验证
# ('acc:', 0.8704691941670365)

# 计算相似度
for line in user_shop_behavior.values:
    for wifi in line[5].split(';'):
        wifi_to_shops[wifi.split('|')[0]][line[1]].append(int(wifi.split('|')[1]))