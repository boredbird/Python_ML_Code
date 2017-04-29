# -*- coding:utf-8 -*-
__author__ = 'chunhui.zhang'
import sys
sys.path.append('E:/Code/Python_ML_Code/JD')
import pandas as pd
import load_data as ld
import time
import numpy as np
from config_params import *

##时段特征合并
user_file = ['trainset4_feature_user','trainset3_feature_user','trainset2_feature_user','trainset1_feature_user','testset_feature_user','predictset_feature_user']
sku_file = ['trainset4_feature_sku','trainset3_feature_sku','trainset2_feature_sku','trainset1_feature_sku','testset_feature_sku','predictset_feature_sku']
cate_file = ['trainset4_feature_cate','trainset3_feature_cate','trainset2_feature_cate','trainset1_feature_cate','testset_feature_cate','predictset_feature_cate']
user_sku_file = ['trainset4_feature_user_sku','trainset3_feature_user_sku','trainset2_feature_user_sku','trainset1_feature_user_sku','testset_feature_user_sku','predictset_feature_user_sku']
interval_file = ['trainset4_feature_interval','trainset3_feature_interval','trainset2_feature_interval','trainset1_feature_interval','testset_feature_interval','predictset_feature_interval']


for i in range(6):
    dataset = ld.load_from_csv(feature_path,[user_sku_file[i],user_file[i],sku_file[i],cate_file[i]])
    # merge
    df = pd.merge(dataset[0],dataset[1],how='left',on='user_id',suffixes=('', '_gbuser'))
    df = pd.merge(df, dataset[2], how='left', on='sku_id', suffixes=('', '_gbsku'))
    df = pd.merge(df, dataset[3], how='left', on='cate', suffixes=('', '_gbcate'))
    # interval feature
    ld.load_into_csv(feature_path, df, file_name=interval_file[i])


##累计属性特征合并
##品牌属性
#    SELECT * FROM  `brand_04`;
##品类属性
#   SELECT * FROM   `cate_02`;
##用户属性
#    SELECT * FROM  `user_03`;
##商品属性
#    SELECT * FROM  `product_02`;
##品类品牌商品排序
# `brand_sku_action_cnt_02`
# `cate_brand_sku_action_cnt_02`
# `cate_sku_action_cnt_02`

# df.columns  = 'acc' + df.columns


#用户在这三个月复购的情况也比较少，所以用户属性慎用
acc_file = ['user_03','brand_04','cate_02','product_02','brand_sku_action_cnt_02','cate_brand_sku_action_cnt_02','cate_sku_action_cnt_02']
df_interval_acc_file = ['trainset4_feature_interval_acc','trainset3_feature_interval_acc','trainset2_feature_interval_acc','trainset1_feature_interval_acc','testset_feature_interval_acc','predictset_feature_interval_acc']
for i in range(6):
    interval_dataset = ld.load_from_csv(feature_path, [interval_file[i]])

    acc_dataset = ld.load_from_csv(acc_feature_path, ['user_03'])
    acc_dataset[0].columns = 'acc_' + acc_dataset[0].columns
    df = pd.merge(interval_dataset[0], acc_dataset[0], how='left', left_on='user_id', right_on='acc_user_id', suffixes=('', '_gbuser'))

    acc_dataset = ld.load_from_csv(acc_feature_path, ['brand_04'])
    acc_dataset[0].columns = 'acc_' + acc_dataset[0].columns
    df = pd.merge(df, acc_dataset[0], how='left', left_on='brand', right_on='acc_brand', suffixes=('', '_gbbrand'))

    acc_dataset = ld.load_from_csv(acc_feature_path, ['cate_02'])
    acc_dataset[0].columns = 'acc_' + acc_dataset[0].columns
    df = pd.merge(df, acc_dataset[0], how='left', left_on='cate', right_on='acc_cate', suffixes=('', '_gbcate'))

    acc_dataset = ld.load_from_csv(acc_feature_path, ['product_02'])
    acc_dataset[0].columns = 'acc_' + acc_dataset[0].columns
    df = pd.merge(df, acc_dataset[0], how='left', left_on='sku_id', right_on='acc_sku_id', suffixes=('', '_gbsku'))

    acc_dataset = ld.load_from_csv(acc_feature_path, ['brand_sku_action_cnt_02'])
    acc_dataset[0].columns = 'acc_' + acc_dataset[0].columns
    df = pd.merge(df, acc_dataset[0], how='left', left_on=['sku_id'], right_on=['acc_sku_id'], suffixes=('', '_gbbs'))

    acc_dataset = ld.load_from_csv(acc_feature_path, ['cate_brand_sku_action_cnt_02'])
    acc_dataset[0].columns = 'acc_' + acc_dataset[0].columns
    df = pd.merge(df, acc_dataset[0], how='left', left_on=['cate', 'brand'], right_on=['acc_cate', 'acc_brand'], suffixes=('', '_gbcb'))

    acc_dataset = ld.load_from_csv(acc_feature_path, ['cate_sku_action_cnt_02'])
    acc_dataset[0].columns = 'acc_' + acc_dataset[0].columns
    df = pd.merge(df, acc_dataset[0], how='left', left_on='sku_id', right_on='acc_sku_id', suffixes=('', '_gbcs'))

    ld.load_into_csv(feature_path, df, file_name=df_interval_acc_file[i])
    print 'Done ',interval_file[i]


##关联标记
# df_interval_acc_file
df_interval_acc_file = ['trainset3_feature_interval_acc','trainset2_feature_interval_acc','trainset1_feature_interval_acc','testset_feature_interval_acc','predictset_feature_interval_acc']
lable_file = ['trainset3_lable','trainset2_lable','trainset1_lable','testset_lable','predictset_lable']
gen_data_file = ['trainset3_kuanbiao','trainset2_kuanbiao','trainset1_kuanbiao','testset_kuanbiao','predictset_kuanbiao']
for i in range(5):
    print 'reading ',df_interval_acc_file[i]
    feature_dataset = pd.read_csv(feature_path + df_interval_acc_file[i] +'.csv')
    print 'Done ',df_interval_acc_file[i]
    lable = ld.load_from_csv(split_data_path, [lable_file[i]])
    lable = lable[0][lable[0]['type'] == 4].loc[:,['user_id','sku_id']]
    lable = lable.drop_duplicates()
    lable['user_id'] = lable['user_id'].astype(int)
    lable['sku_id'] = lable['sku_id'].astype(int)
    lable['mark'] = 1

    feature_dataset['user_id'] = feature_dataset['user_id'].astype(int)
    feature_dataset['sku_id'] = feature_dataset['sku_id'].astype(int)
    print sum(lable['mark'])
    df = pd.merge(feature_dataset, lable, how='left', on=['user_id','sku_id'])
    # release memory size
    feature_dataset = []
    lable = []
    df['if_order_cnt'] = df['order_cnt'] > 0
    ld.load_into_csv(gen_data_path, df, file_name=gen_data_file[i])
    print df.shape
    print sum(df['order_cnt'] > 0)
    print sum(df['mark']==1)
    df = []
    print 'Done ', gen_data_file[i]

    # feature_dataset.dtypes[feature_dataset.dtypes == 'object'] = pd.Series(['float'] * 24)
    # pd.set_option('max_info_columns', 500)
    # a = feature_dataset.dtypes
    # a[a == 'object'] = pd.Series(['int'] * 24)

"""
reading  trainset3_feature_interval_acc
Done  trainset3_feature_interval_acc
reading: trainset3_lable2017-04-29 23:57:22
done: trainset3_lable2017-04-29 23:57:26
2801
(1941958, 132)
22198
1207
Done  trainset3_kuanbiao
reading  trainset2_feature_interval_acc
D:\Program Files\Anaconda2\lib\site-packages\IPython\core\interactiveshell.py:2717: DtypeWarning: Columns (32,38,48,49,50,51,52,53,54,55,56,57,58,59,84,87) have mixed types. Specify dtype option on import or set low_memory=False.
  interactivity=interactivity, compiler=compiler, result=result)
Done  trainset2_feature_interval_acc
reading: trainset2_lable2017-04-29 23:59:57
done: trainset2_lable2017-04-30 00:00:01
3152
(2050643, 132)
22754
1384
Done  trainset2_kuanbiao
reading  trainset1_feature_interval_acc
Done  trainset1_feature_interval_acc
reading: trainset1_lable2017-04-30 00:03:38
done: trainset1_lable2017-04-30 00:03:47
2826
(2066975, 132)
22763
1234
Done  trainset1_kuanbiao
reading  testset_feature_interval_acc
Done  testset_feature_interval_acc
reading: testset_lable2017-04-30 00:06:44
done: testset_lable2017-04-30 00:06:49
4942
(2008781, 132)
22374
2689
Done  testset_kuanbiao
reading  predictset_feature_interval_acc
D:\Program Files\Anaconda2\lib\site-packages\IPython\core\interactiveshell.py:2717: DtypeWarning: Columns (32,33,34,35,36,37,38,48,49,50,51,52,53,54,55,56,57,58,59,79,80,81,82,83,84,87) have mixed types. Specify dtype option on import or set low_memory=False.
  interactivity=interactivity, compiler=compiler, result=result)
Done  predictset_feature_interval_acc
reading: predictset_lable2017-04-30 00:09:14
done: predictset_lable2017-04-30 00:09:14
0
(1860498, 132)
20753
0
Done  predictset_kuanbiao
"""