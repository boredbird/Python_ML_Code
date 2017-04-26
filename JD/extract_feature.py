# -*- coding:utf-8 -*-
__author__ = 'chunhui.zhang'
import sys
sys.path.append('E:/Code/Python_ML_Code/JD')
from config_params import *
import load_data as ld
import numpy as np

def extract_user_action_feature(path,file):
    dataset = ld.load_from_csv(path,file)
    file_cnt = dataset.__len__()
    for i in range(file_cnt):
        # dataset[i]
        print i

    return

extract_user_action_feature(split_data_path,['trainset4_feature','trainset3_feature',])

dataset = ld.load_from_csv(split_data_path,['trainset4_feature','trainset3_feature',])




# user_set = pd.DataFrame({'user_set':user_set}).iloc[:,0]
user_set.size
75030
len(set(dataset[0][dataset[0]['type'] == 1]['user_id']))
73794
##也就是说并不是每个人都有点击行为
##set 转为 dataframe
user_set = set(dataset[0]['user_id'])
user_set = pd.DataFrame({'user_id':list(user_set),})
user_set['user_id'] = user_set['user_id'].astype(int)


type_list = ['browse_cnt','shoppingcar_in_cnt','shoppingcar_out_cnt','order_cnt','follow_cnt','click_cnt']
g = dataset[0].groupby('user_id')
gb = g.apply(lambda x: x[x['type'] == 1].count())
gb['user_id'] = gb['user_id'].astype(int)

a = pd.merge(user_set, gb, how='left', on='user_id')


gb = gb.drop(['sku_id','time','model_id','type','cate','brand'],axis=1)
gb = gb.reset_index()
gb.columns

ld.load_into_csv(feature_path, a, file_name='a')