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

wifi_id = []
min_time = []
max_time = []
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


eval_counter = defaultdict(lambda: 0)
for line in evalset.values:
    for wifi in line[6].split(';'):
        eval_counter[wifi.split('|')[0]] +=1


print sorted([var for var in eval_counter.items()],key=lambda x:int(x[1]),reverse=True)[:100]
"""
eval_counter_list.__len__()
Out[61]:
224255
"""
eval_counter_selected = [var for var in eval_counter.items() if var[1]>1] # 145207

eval_wifi_time = defaultdict(lambda: [])
for line in evalset.values:
    for wifi in line[6].split(';'):
        eval_wifi_time[wifi.split('|')[0]].append(line[3])

a = [[k,min(v),max(v)] for k,v in eval_wifi_time.items()] # 224254
b = pd.DataFrame(a,columns=['wifi_id','min_time','max_time'])
day_diff = [(parser.parse(var[2])-parser.parse(var[1])).days for var in b.values]
b['day_diff'] = day_diff
b.groupby(day_diff).count()
"""
    wifi_id  min_time  max_time  day_diff
0    123160    123160    123160    123160
1      3548      3548      3548      3548
2      2980      2980      2980      2980
3      2890      2890      2890      2890
4      2987      2987      2987      2987
5      3210      3210      3210      3210
6      3878      3878      3878      3878
7      4819      4819      4819      4819
8      5320      5320      5320      5320
9      5525      5525      5525      5525
10     6616      6616      6616      6616
11     9478      9478      9478      9478
12    15589     15589     15589     15589
13    34254     34254     34254     34254
"""
