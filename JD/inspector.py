# -*- coding:utf-8 -*-
__author__ = 'maomaochong'
import sys
sys.path.append('E:/Code/Python_ML_Code/JD')
import pandas as pd
import load_data as ld
import time
import numpy as np
from config_params import *

"""
# create table inspector_03
# as
# select t1.`user_id`,t1.`sku_id`
# ,MAX(TO_DAYS(t2.TIME)) as max_time
# ,MIN(TO_DAYS(t2.TIME)) as min_time
# ,MAX(TO_DAYS(t2.TIME))-MIN(TO_DAYS(t2.TIME)) AS time_diff
# from inspector_02 t1
# inner join jdata_action_all t2
# on t1.user_id=t2.user_id and t1.sku_id=t2.sku_id
# group by t1.`user_id`,t1.`sku_id`;
"""
raw_data_path = 'E:/Code/Python_ML_Code/JD/raw_data/'

jdata_action_all = ld.load_from_csv(raw_data_path,['JData_Action_ALL',])
inspector_02 = ld.load_from_mysql('inspector_02')

print 'reading: ' +  time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
result = pd.merge(inspector_02, jdata_action_all[0], how='inner', on=['user_id', 'sku_id'])
print 'done: ' +  time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
ld.load_into_csv(feature_path, result, file_name='inspector_03')

# result1 = result.loc[:5,]
# ld.load_into_mysql(result1,'inspector_03')

#
#
# result01 = result.groupby('user_id','sku_id').agg({'time_y':np.max,'time_y':np.min})


# result = load_data.load_from_csv(raw_data_path,['user_sku_max_min_time',])
# result[0]['time_y'] = result[0]['time_y'].astype(np.datetime64)
# result01 = result[0].groupby(['user_id','sku_id'])['time_y'].agg(lambda arr:arr.max()-arr.min())
#
# import matplotlib.pyplot as plt
# plt.plot(result01[2].days)

# pd.to_datetime('2016-04-15')-pd.to_datetime('2016-04-10')
# pd.bdate_range(end='2016-04-15',periods=5)


###时间日期相减
# import datetime
# d2 = '2016-04-15' + datetime.timedelta(-1)
# pd.to_datetime('2016-04-15')+ datetime.timedelta(-1)
# datetime.date.today() + datetime.timedelta(days=10)
#
# from pandas.tseries.offsets import Day
# days=Day(30)
# import datetime
# now=datetime.datetime.now()
# now-days


dataset = ld.load_from_csv(feature_path,['user_02','user_01',])
dataset[0][dataset[0]['order_cnt']>0].info()
#行为表中的用户数:104301
#下单用户数：26919
#用户表中的用户数：105321
#action type 1 数:305321



dataset = ld.load_from_csv(split_data_path,['trainset4_feature',])

set(dataset[0]['dt'])
Out[6]:
{'2016-02-01',
 '2016-02-08',
 '2016-02-15',
 '2016-02-22',
 '2016-02-29',
 '2016-03-07',
 '2016-03-14',
 '2016-03-21',
 '2016-03-28',
 '2016-04-04',
 '2016-04-11',
 '2016-04-15'}

feature_file = ['trainset4_feature','trainset3_feature','trainset2_feature','trainset1_feature','testset_feature','predictset_feature']
lable_file = ['trainset4_lable','trainset3_lable','trainset2_lable','trainset1_lable','testset_lable','predictset_lable']

for i in range(6):
    fa = feature_file[i]
    fb = lable_file[i]
    dataseta = ld.load_from_csv(split_data_path,[fa,])
    datasetb = ld.load_from_csv(split_data_path,[fb,])
    user_seta =  set(dataseta[0]['user_id'])
    user_setb = set(datasetb[0]['user_id'])
    print fa,'  ',fb
    print len(user_seta)
    print len(user_setb)
    print len(user_seta&user_setb)

"""
trainset4_feature    trainset4_lable
75030
50571
43156
trainset3_feature    trainset3_lable
91380
52636
49163
trainset2_feature    trainset2_lable
94537
50277
47451
trainset1_feature    trainset1_lable
95614
50733
47893
testset_feature    testset_lable
96576
52092
49988
predictset_feature    predictset_lable
94359
"""

dataset = ld.load_from_csv(raw_data_path,['JData_Action_ALL',])
"""
dataset = ld.load_from_csv(raw_data_path,['JData_Action_ALL',])
reading: JData_Action_ALL2017-04-30 12:31:11
done: JData_Action_ALL2017-04-30 12:31:53
df = dataset[0][dataset[0]['type'] == 4]
sum(dataset[0]['type'] == 4)
Out[5]:
48252
df = df.loc[:,['user_id','sku_id']]
df = df.drop_duplicates()
df.info()
<class 'pandas.core.frame.DataFrame'>
Int64Index: 46607 entries, 351 to 50601111
Data columns (total 2 columns):
user_id    46607 non-null float64
sku_id     46607 non-null int64
dtypes: float64(1), int64(1)
memory usage: 1.1 MB
"""

df['user_id'] = df['user_id'].astype(int)
df['sku_id'] = df['sku_id'].astype(int)
dataset[0]['user_id'] = dataset[0]['user_id'].astype(int)
dataset[0]['sku_id'] = dataset[0]['sku_id'].astype(int)

dataset[0] = pd.merge(dataset[0], df, how='inner', on=['user_id', 'sku_id'])
##4893684
ld.load_into_csv(feature_path, dataset[0], file_name='user_sku_order_action')



dataset = ld.load_from_csv(raw_data_path,['JData_Action_ALL',])
dataset = ld.load_from_csv(raw_data_path,['JData_Action_ALL',])