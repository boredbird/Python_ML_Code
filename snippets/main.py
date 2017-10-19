# -*- coding:utf-8 -*-
__author__ = 'chunhui.zhang'
import sys
sys.path.append(r'E:\Code\ScoreCard')
from config_params import  *
import load_data as ld
import change_feature_dtype as cfd
import data_binning as bin
import calculate_iv as ci
import woe_transformation as wt
from sklearn import linear_model

#加载数据
dataset =ld.load_from_csv(rawdata_path,['pos_model_var_tbl_train',])
dataset_train = dataset[0]
dataset_train.columns = [col.split('.')[-1] for col in dataset_train.columns]
dataset_len = len(dataset_train)

#改变数据类型
cfd.change_feature_dtype(dataset_train)

########分箱处理
bin_var_list = config[config['is_tobe_bin']==1]['var_name']
for var in bin_var_list:
    try:
        print var
        #计算最优分隔点
        split = bin.binning_data_split(dataset_train,var)
        # print split

        #合并小组别
        split = bin.check_point(dataset_train,var,split,dataset_len)
        # print split

        dic = ci.calculate_split_iv(dataset_train, var, split)
        print dic

        # 跟进分隔点区间将连续值替换成分类变量
        dataset_train[var] = bin.c2d(dataset_train[var], split)

    except Exception, e:
        print 'ERROR DATA BIN:',var

#计算IV和WOE替换
candidate_var_list = config[config['is_candidate']==1]['var_name']
iv_list = pd.Series()
for var in candidate_var_list:
    #计算每个变量的IV值
    try:
        # iv_list[var] = ci.calculate_iv(dataset_train,var)
        # print var,' : ', iv_list[var]

        # WOE变换
        dataset_train[var] = wt.woe_transformation(dataset_train,var)
        print 'SUCCESS DONE WOE TRANS: ',var
    except Exception, e:
        print 'ERROR CALCULATE IV: ',var


# dataset_train = pd.read_csv('E:\ScoreCard\gendata\dataset_train_woe.csv')
# dataset_train.to_csv('E:\ScoreCard\gendata\dataset_train_woe.csv', index=False)

var = 'pos_due_periods_ratio'
a = dataset_train.loc[:,[var,'target']]
b = a.groupby([var, 'target'])['target'].count()
print b

dataset_train[var] = wt.woe_transformation(dataset_train,var)
print set(dataset_train[var])


# dataset_train[var].fillna('missing')
# sum(pd.isnull(dataset_train[var]))
# dataset_train[var][pd.isnull(dataset_train[var])] = 'misssing'




