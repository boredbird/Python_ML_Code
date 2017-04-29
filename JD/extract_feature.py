# -*- coding:utf-8 -*-
__author__ = 'chunhui.zhang'
import sys
sys.path.append('E:/Code/Python_ML_Code/JD')
from config_params import *
import load_data as ld
import numpy as np

# user_set = pd.DataFrame({'user_set':user_set}).iloc[:,0]
# user_set.size
# 75030
# len(set(dataset[0][dataset[0]['type'] == 1]['user_id']))
# 73794
##也就是说并不是每个人都有点击行为

##################################group by user_id##############################
feature_file = ['trainset4_feature','trainset3_feature','trainset2_feature','trainset1_feature','testset_feature','predictset_feature']

for file in feature_file:
    adict = locals()

    dataset = ld.load_from_csv(split_data_path,[file,])
    ##set 转为 dataframe
    user_set = set(dataset[0]['user_id'])
    user_set = pd.DataFrame({'user_id':list(user_set),})
    user_set['user_id'] = user_set['user_id'].astype(int)

    ##type
    type_list = ['browse_cnt','shoppingcar_in_cnt','shoppingcar_out_cnt','order_cnt','follow_cnt','click_cnt']
    g = dataset[0].groupby('user_id')
    adict['df_%s' % (file)] = user_set
    for i in range(6):
        #group by
        gb = g.apply(lambda x: x[x['type'] == i+1].count())
        gb = gb.drop(['sku_id', 'time', 'model_id', 'type', 'cate', 'brand'], axis=1)
        gb.columns = list([type_list[i],])
        gb = gb.reset_index()
        gb['user_id'] = gb['user_id'].astype(int)
        #merge
        adict['df_%s' % (file)] = pd.merge(adict['df_%s' % (file)], gb, how='left', on='user_id')
        print 'done ',file,' ',i,' ',type_list[i]

    #merge all type done
    ld.load_into_csv(feature_path, adict['df_%s' % (file)], file_name=file+'_user')
    print 'Done ', file, '!!!'



##################################lable##############################
lable_file = ['trainset4_lable','trainset3_lable','trainset2_lable','trainset1_lable','testset_lable','predictset_lable']

for file in lable_file:
    dataset = ld.load_from_csv(split_data_path, [file, ])
    # select user_id,sku_id where type==4
    df = dataset[0][dataset[0]['type'] == 4].loc[:,['user_id','sku_id']]
    df['user_id'] = df['user_id'].astype(int)
    df = df.drop_duplicates()
    ld.load_into_csv(feature_path, df, file_name=file)
    print 'Done ', file, '!!!'



##################################group by sku_id##############################
feature_file = ['trainset4_feature','trainset3_feature','trainset2_feature','trainset1_feature','testset_feature','predictset_feature']

for file in feature_file:
    adict = locals()

    dataset = ld.load_from_csv(split_data_path,[file,])
    ##set 转为 dataframe
    sku_set = set(dataset[0]['sku_id'])
    sku_set = pd.DataFrame({'sku_id':list(sku_set),})
    sku_set['sku_id'] = sku_set['sku_id'].astype(int)

    ##type
    type_list = ['browse_cnt','shoppingcar_in_cnt','shoppingcar_out_cnt','order_cnt','follow_cnt','click_cnt']
    g = dataset[0].groupby('sku_id')
    adict['df_%s' % (file)] = sku_set
    for i in range(6):
        #group by
        gb = g.apply(lambda x: x[x['type'] == i+1].count())
        gb = gb.drop(['sku_id', 'time', 'model_id', 'type', 'cate', 'brand'], axis=1)
        gb.columns = list([type_list[i],])
        gb = gb.reset_index()
        gb['sku_id'] = gb['sku_id'].astype(int)
        #merge
        adict['df_%s' % (file)] = pd.merge(adict['df_%s' % (file)], gb, how='left', on='sku_id')
        print 'done ',file,' ',i,' ',type_list[i]

    #merge all type done
    ld.load_into_csv(feature_path, adict['df_%s' % (file)], file_name=file+'_sku')
    print 'Done ', file, '!!!'


##################################group by cate##############################
'trainset4_feature','trainset3_feature','trainset2_feature','trainset1_feature',
feature_file = ['testset_feature','predictset_feature']

for file in feature_file:
    adict = locals()

    dataset = ld.load_from_csv(split_data_path,[file,])
    ##set 转为 dataframe
    cate_set = set(dataset[0]['cate'])
    cate_set = pd.DataFrame({'cate':list(cate_set),})
    cate_set['cate'] = cate_set['cate'].astype(int)

    ##type
    type_list = ['browse_cnt','shoppingcar_in_cnt','shoppingcar_out_cnt','order_cnt','follow_cnt','click_cnt']
    g = dataset[0].groupby('cate')
    adict['df_%s' % (file)] = cate_set
    for i in range(6):
        #group by
        gb = g.apply(lambda x: x[x['type'] == i+1].count())
        gb = gb.drop(['sku_id', 'time', 'model_id', 'type', 'cate', 'brand'], axis=1)
        gb.columns = list([type_list[i],])
        gb = gb.reset_index()
        gb['cate'] = gb['cate'].astype(int)
        #merge
        adict['df_%s' % (file)] = pd.merge(adict['df_%s' % (file)], gb, how='left', on='cate')
        print 'done ',file,' ',i,' ',type_list[i]

    #merge all type done
    ld.load_into_csv(feature_path, adict['df_%s' % (file)], file_name=file+'_cate')
    print 'Done ', file, '!!!'



##################################group by user_id sku_id##############################
feature_file = ['trainset4_feature','trainset3_feature','trainset2_feature','trainset1_feature','testset_feature','predictset_feature']
##type
type_list = ['browse_cnt', 'shoppingcar_in_cnt', 'shoppingcar_out_cnt', 'order_cnt', 'follow_cnt', 'click_cnt']

for file in feature_file:
    dataset = ld.load_from_csv(split_data_path,[file,])

    user_sku_set = dataset[0].loc[:,['user_id','sku_id','cate','brand']].drop_duplicates()
    user_sku_set['user_id'] = user_sku_set['user_id'].astype(int)
    user_sku_set['sku_id'] = user_sku_set['sku_id'].astype(int)

    for i in range(6):
        print i
        #group by
        a = dataset[0][dataset[0]['type'] == i + 1]
        df = a.groupby(['user_id','sku_id']).count()
        #tidy data type
        df = df.drop(['model_id', 'type','cate','brand'], axis=1)
        df.columns = list([type_list[i],])
        df = df.reset_index()
        df['user_id'] = df['user_id'].astype(int)
        df['sku_id'] = df['sku_id'].astype(int)
        #merge
        user_sku_set = pd.merge(user_sku_set, df, how='left', on=['user_id','sku_id'])

    ld.load_into_csv(feature_path, user_sku_set, file_name=file+'_user_sku')
    print 'Done ', file, '!!!'

