#-*- coding:utf-8 -*-

"""
统计wifi出现次数与时间的关系
"""
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

wifi_time = defaultdict(lambda: [])
for line in user_shop_behavior.values:
    for wifi in line[5].split(';'):
        wifi_time[wifi.split('|')[0]].append(line[2])

a = [[k,min(v),max(v)] for k,v in wifi_time.items()] # 399679

b = pd.DataFrame(a,columns=['wifi_id','min_time','max_time'])
print b[b['max_time'] <'2017-08-31'].shape # 332557
print b[b['max_time'] <'2017-08-30'].shape # 308682

import time
from dateutil import parser
day_diff = [(parser.parse(var[2])-parser.parse(var[1])).days for var in b.values]
b['day_diff'] = day_diff

b.groupby(day_diff).count()
"""
b.groupby(day_diff).count()
Out[44]:
    wifi_id  min_time  max_time  day_diff
0    260137    260137    260137    260137
1      3766      3766      3766      3766
2      3228      3228      3228      3228
3      2803      2803      2803      2803
4      2496      2496      2496      2496
5      2510      2510      2510      2510
6      2680      2680      2680      2680
7      2777      2777      2777      2777
8      2462      2462      2462      2462
9      2316      2316      2316      2316
10     2300      2300      2300      2300
11     2230      2230      2230      2230
12     2370      2370      2370      2370
13     2419      2419      2419      2419
14     2579      2579      2579      2579
15     2380      2380      2380      2380
16     2401      2401      2401      2401
17     2497      2497      2497      2497
18     2525      2525      2525      2525
19     2695      2695      2695      2695
20     2914      2914      2914      2914
21     3185      3185      3185      3185
22     3438      3438      3438      3438
23     3681      3681      3681      3681
24     4146      4146      4146      4146
25     4926      4926      4926      4926
26     5867      5867      5867      5867
27     6738      6738      6738      6738
28     8710      8710      8710      8710
29    13749     13749     13749     13749
30    32754     32754     32754     32754
"""

valid_wifi = b[b['day_diff']>28] # 32754

# valid_wifi = b[b['day_diff']>28]  46503

# 训练规则
wifi_to_shops = defaultdict(lambda : defaultdict(lambda :0)) # 34766
valid_wifi.index = valid_wifi['wifi_id']

print time.asctime(time.localtime(time.time()))
for line in user_shop_behavior.values:
    try:
        wifi = sorted([wifi.split('|') for wifi in line[5].split(';') if wifi.split('|')[0] in  valid_wifi.index
                       ],key=lambda x:int(x[1]),reverse=True)[0]
        wifi_to_shops[wifi[0]][line[1]] = wifi_to_shops[wifi[0]][line[1]] + 1
    except:
        continue

print time.asctime(time.localtime(time.time()))

"""
wifi_to_shops.items().__len__()
Out[7]:
34766

wifi_to_shops.items().__len__()
Out[11]:
27327
"""

# 线下验证
right_count = 0
for line in user_shop_behavior.values:
    try:
        wifi = sorted([wifi.split('|') for wifi in line[5].split(';') if wifi.split('|')[0] in valid_wifi.index
                       ],key=lambda x:int(x[1]),reverse=True)[0]
        counter = defaultdict(lambda : 0)
        for k,v in wifi_to_shops[wifi[0]].items():
            counter[k] += v
        pred_one = sorted(counter.items(),key=lambda x:x[1],reverse=True)[0][0]
        if pred_one == line[1]:
            right_count += 1
    except:
        continue
print('acc:',right_count*1.0/len(user_shop_behavior))
# ('acc:', 0.8305303532905981)
# ('acc:', 0.803246881631613)

# 验证wifi_to_shops是否全部覆盖了所有的shop_id
s = []
for wifi in wifi_to_shops.values():
    for var in wifi.items():
        s.append(var[0])

s_unique = np.unique(s)

# s_unique.__len__()
# 8151


wifi_to_shops = defaultdict(lambda : defaultdict(lambda :0))
for line in user_shop_behavior.values:
    wifi = sorted([wifi.split('|') for wifi in line[5].split(';')],key=lambda x:int(x[1]),reverse=True)[0]
    wifi_to_shops[wifi[0]][line[1]] = wifi_to_shops[wifi[0]][line[1]] + 1

s = []
for wifi in wifi_to_shops.values():
    for var in wifi.items():
        s.append(var[0])

s_unique = np.unique(s)

# s_unique.__len__()
# 8477

"""
查看wifi已连接
"""
wifi_to_shops = defaultdict(lambda : defaultdict(lambda :0))
for line in user_shop_behavior.values:
    # 864
    try:
        wifi = sorted([wifi.split('|') for wifi in line[5].split(';') if wifi.split('|')[2]=='true'],key=lambda x:int(x[1]),reverse=True)[0]
        wifi_to_shops[wifi[0]][line[1]] = wifi_to_shops[wifi[0]][line[1]] + 1
    except:
        continue
"""
wifi_to_shops.items().__len__()
Out[43]:
39183
"""

# 线下验证
right_count = 0
wifi_to_shops_series = pd.Series(wifi_to_shops.keys(),index=wifi_to_shops.keys())
for line in user_shop_behavior.values:
    try:
        wifi = sorted([wifi.split('|') for wifi in line[5].split(';')
                        if wifi.split('|')[0] in wifi_to_shops_series.index
                       ],key=lambda x:int(x[1]),reverse=True)[0]
        counter = defaultdict(lambda : 0)
        for k,v in wifi_to_shops[wifi[0]].items():
            counter[k] += v
        pred_one = sorted(counter.items(),key=lambda x:x[1],reverse=True)[0][0]
        if pred_one == line[1]:
            right_count += 1
    except:
        continue
print('acc:',right_count*1.0/len(user_shop_behavior))
# ('acc:', 0.6030676221315184)

# 计算有wifi连接的那部分的ACC
# 线下验证
right_count = 0
total_count = 0
wifi_to_shops_series = pd.Series(wifi_to_shops.keys(),index=wifi_to_shops.keys())
for line in user_shop_behavior.values:
    try:
        wifi = sorted([wifi.split('|') for wifi in line[5].split(';')
                       if wifi.split('|')[0] in wifi_to_shops_series.index  and wifi.split('|')[2]=='true'
                       ],key=lambda x:int(x[1]),reverse=True)[0]
        counter = defaultdict(lambda : 0)
        for k,v in wifi_to_shops[wifi[0]].items():
            counter[k] += v
        pred_one = sorted(counter.items(),key=lambda x:x[1],reverse=True)[0][0]

        total_count += 1
        if pred_one == line[1]:
            right_count += 1
    except:
        continue
print('acc:',right_count*1.0/total_count)
# ('acc:', 0.8707236955547125)

"""
查看wifi已连接有效wifi
"""
# 有效wifi
wifi_time = defaultdict(lambda: [])
for line in user_shop_behavior.values:
    for wifi in line[5].split(';'):
        wifi_time[wifi.split('|')[0]].append(line[2])

a = [[k,min(v),max(v)] for k,v in wifi_time.items()] # 399679
b = pd.DataFrame(a,columns=['wifi_id','min_time','max_time'])

day_diff = [(parser.parse(var[2])-parser.parse(var[1])).days for var in b.values]
b['day_diff'] = day_diff
valid_wifi = b[b['day_diff']>28]
valid_wifi.index = valid_wifi['wifi_id'] # 46503

# 查看wifi已连接有效wifi
wifi_to_shops = defaultdict(lambda : defaultdict(lambda :0)) # 18749
for line in user_shop_behavior.values:
    try:
        wifi = sorted([wifi.split('|') for wifi in line[5].split(';')
                       if wifi.split('|')[2]=='true' and wifi.split('|')[0] in valid_wifi.index
                       ],key=lambda x:int(x[1]),reverse=True)[0]
        wifi_to_shops[wifi[0]][line[1]] = wifi_to_shops[wifi[0]][line[1]] + 1
    except:
        continue

# 计算有wifi连接有效wifi的那部分的ACC
# 线下验证
right_count = 0
total_count = 0
wifi_to_shops_series = pd.Series(wifi_to_shops.keys(),index=wifi_to_shops.keys())
for line in user_shop_behavior.values:
    try:
        wifi = sorted([wifi.split('|') for wifi in line[5].split(';')
                       if wifi.split('|')[0] in wifi_to_shops_series.index  and wifi.split('|')[2]=='true'
                       ],key=lambda x:int(x[1]),reverse=True)[0]
        counter = defaultdict(lambda : 0)
        for k,v in wifi_to_shops[wifi[0]].items():
            counter[k] += v
        pred_one = sorted(counter.items(),key=lambda x:x[1],reverse=True)[0][0]

        total_count += 1
        if pred_one == line[1]:
            right_count += 1
    except:
        continue
print('acc:',right_count*1.0/total_count)
# ('acc:', 0.8606539678861639)
"""
right_count
Out[62]:
163427
total_count
Out[63]:
189887
"""


"""
预测集
"""
eval_wifi_to_shops = defaultdict(lambda : defaultdict(lambda :0)) # 23913
for line in evalset.values:
    try:
        wifi = sorted([wifi.split('|') for wifi in line[6].split(';')
                       if wifi.split('|')[2]=='true'
                       ],key=lambda x:int(x[1]),reverse=True)[0]
        eval_wifi_to_shops[wifi[0]][line[1]] = eval_wifi_to_shops[wifi[0]][line[1]] + 1
    except:
        continue

s = []
for wifi in eval_wifi_to_shops.values():
    for var in wifi.items():
        s.append(var[0])

s_unique = np.unique(s) # 67412

row_id = []
preds = []
for line in evalset.values:
    try:
        wifi = sorted([wifi.split('|') for wifi in line[6].split(';')
                       if wifi.split('|')[0] in wifi_to_shops_series.index and wifi.split('|')[2]=='true'
                       ],key=lambda x:int(x[1]),reverse=True)[0]
        counter = defaultdict(lambda : 0)
        for k,v in wifi_to_shops[wifi[0]].items():
            counter[k] += v
        pred_one = sorted(counter.items(),key=lambda x:x[1],reverse=True)[0][0]
        row_id.append(line[0])
        preds.append(pred_one)
    except:
        continue


result = pd.DataFrame({'row_id':row_id,'shop_id':preds})
result.to_csv(r'E:\output\submit\BDCI_first_round_alg13_submit01.csv.csv',index=None)
# score:

"""
分mall查看准确率
"""
# 线下分mall验证
# shop_info.loc['s_26',]['mall_id']
mall_right_agg = defaultdict(lambda :0)
mall_cnt_agg = defaultdict(lambda :0)
for line in user_shop_behavior.values:
    try:
        wifi = sorted([wifi.split('|') for wifi in line[5].split(';') if wifi.split('|')[0] in valid_wifi.index
                       ],key=lambda x:int(x[1]),reverse=True)[0]
        counter = defaultdict(lambda : 0)
        for k,v in wifi_to_shops[wifi[0]].items():
            counter[k] += v
        pred_one = sorted(counter.items(),key=lambda x:x[1],reverse=True)[0][0]
        mall_cnt_agg[line[6]] += 1
        if pred_one == line[1]:
            mall_right_agg[line[6]] += 1
    except:
        continue

mall_acc = defaultdict(lambda :0)
for var in mall_cnt_agg.items():
    mall_acc[var[0]] = mall_right_agg[var[0]]*1.0/mall_cnt_agg[var[0]]
