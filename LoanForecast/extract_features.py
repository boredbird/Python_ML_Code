# -*- coding:utf-8 -*-
__author__ = 'maomaochong'
import pandas as pd
import  numpy as np
import datetime
import time

"""
dataset split:
           dataset1 (113640): [features] 20160803~20160901,[target] 20160902~20161001
           dataset2 (113640): [features] 20160902~20161001,[target] 20161002~20161031
           dataset3 (113640): [features] 20161002~20161031,[target] 20161101~20161130
predictionset (113640): [features] 20161101~20161130,[target] 20161201~20161231

"""
raw_dataset_path = 'E:\\LoanForecast\\RawData\\'
gen_dataset_path = 'E:\\LoanForecast\\GenData\\'

######################################### get_user_order_features() ###################################
# user order features
# group by uid
dataset_order = pd.read_csv(raw_dataset_path+'t_order.csv')

dataset_order_set1 = dataset_order[(dataset_order['buy_time']>'2016-08-03') & (dataset_order['buy_time']<'2016-09-02')]

user_order_features = {}

df_tmp = dataset_order_set1[['uid','buy_time']]
user_order_features['uid_order_cnt'] = df_tmp.groupby(['uid']).count().iloc[:,0]

df_tmp = dataset_order_set1[['uid','buy_time']]
user_order_features['uid_buy_time_min'] = df_tmp.groupby(['uid']).agg('min')['buy_time']

df_tmp = dataset_order_set1[['uid','buy_time']]
user_order_features['uid_buy_time_max'] = df_tmp.groupby(['uid']).agg('max')['buy_time']

df_tmp = dataset_order_set1[['uid','price']]
user_order_features['uid_price_sum'] = df_tmp.groupby(['uid']).sum()['price']

df_tmp = dataset_order_set1[['uid','price']]
user_order_features['uid_price_avg'] = df_tmp.groupby(['uid']).agg('mean')['price']

df_tmp = dataset_order_set1[['uid','discount']]
user_order_features['uid_discount_sum'] = df_tmp.groupby(['uid']).sum()['discount']

df_tmp = dataset_order_set1[['uid','discount']]
user_order_features['uid_price_avg'] = df_tmp.groupby(['uid']).agg('mean')['discount']

df_tmp = dataset_order_set1[['uid','qty']]
user_order_features['uid_qty_sum'] = df_tmp.groupby(['uid']).sum()['qty']

df_tmp = dataset_order_set1[['uid','qty']]
user_order_features['uid_qty_avg'] = df_tmp.groupby(['uid']).agg('mean')['qty']

df_tmp = dataset_order_set1[['uid','qty']]
user_order_features['uid_qty_min'] = df_tmp.groupby(['uid']).agg('min')['qty']

df_tmp = dataset_order_set1[['uid','qty']]
user_order_features['uid_qty_max'] = df_tmp.groupby(['uid']).agg('max')['qty']

df_tmp = dataset_order_set1[['uid','qty']]
user_order_features['uid_qty_cnt'] = df_tmp.groupby(['uid']).agg({'qty': pd.Series.nunique}).qty

df_tmp = dataset_order_set1[['uid','cate_id']]
user_order_features['uid_cate_cnt'] = df_tmp.groupby(['uid']).agg({'cate_id': pd.Series.nunique}).cate_id

user_order_features_on_uid = pd.DataFrame(user_order_features)
datetime.datetime(user_order_features['uid_buy_time_max']) - datetime.datetime(user_order_features['uid_buy_time_min'])

a = user_order_features['uid_buy_time_max'].apply(lambda s:time.strptime(s,"%Y-%m-%d"))
b = user_order_features['uid_buy_time_min'].apply(lambda s:time.strptime(s,"%Y-%m-%d"))
user_order_features['uid_buy_time_max'] = (datetime.datetime(*a[:3]) - datetime.datetime(*b[:3])).days

# user order features on cate_id
# group by uid,cate_id
user_order_features = {}

df_tmp = dataset_order_set1[['uid','cate_id','buy_time']]
user_order_features['uid_cate_order_cnt'] = df_tmp.groupby(['uid','cate_id']).count().iloc[:,0]

df_tmp = dataset_order_set1[['uid','cate_id','price']]
user_order_features['uid_cate_price_sum'] = df_tmp.groupby(['uid','cate_id']).sum()['price']

df_tmp = dataset_order_set1[['uid','cate_id','price']]
user_order_features['uid_cate_price_sum'] = df_tmp.groupby(['uid','cate_id']).sum()['price']

df_tmp = dataset_order_set1[['uid','cate_id','discount']]
user_order_features['uid_cate_discount_sum'] = df_tmp.groupby(['uid','cate_id']).sum()['discount']

df_tmp = dataset_order_set1[['uid','cate_id','qty']]
user_order_features['uid_cate_qty_sum'] = df_tmp.groupby(['uid','cate_id']).sum()['qty']

df_tmp = dataset_order_set1[['uid','cate_id','qty']]
user_order_features['uid_cate_qty_avg'] = df_tmp.groupby(['uid','cate_id']).agg('mean')['qty']

df_tmp = dataset_order_set1[['uid','cate_id','price']]
user_order_features['uid_cate_price_avg'] = df_tmp.groupby(['uid','cate_id']).agg('mean')['price']

df_tmp = dataset_order_set1[['uid','cate_id','discount']]
user_order_features['uid_cate_discount_avg'] = df_tmp.groupby(['uid','cate_id']).agg('mean')['discount']

df_tmp = dataset_order_set1[['uid','cate_id','qty']]
user_order_features['uid_cate_qty_min'] = df_tmp.groupby(['uid','cate_id']).agg('min')['qty']

df_tmp = dataset_order_set1[['uid','cate_id','buy_time']]
user_order_features['uid_cate_buy_time_min'] = df_tmp.groupby(['uid','cate_id']).agg('min')['buy_time']

df_tmp = dataset_order_set1[['uid','cate_id','qty']]
user_order_features['uid_cate_qty_max'] = df_tmp.groupby(['uid','cate_id']).agg('max')['qty']

df_tmp = dataset_order_set1[['uid','cate_id','buy_time']]
user_order_features['uid_cate_buy_time_max'] = df_tmp.groupby(['uid','cate_id']).agg('max')['buy_time']

df_tmp = dataset_order_set1[['uid','cate_id','qty']]
user_order_features['uid_cate_qty_cnt'] = df_tmp.groupby(['uid','cate_id']).agg({'qty': pd.Series.nunique}).qty

user_order_features_on_uid_cate = pd.DataFrame(user_order_features)
user_order_features_on_uid_cate = user_order_features_on_uid_cate.unstack(level=-1)

# user_order_features['uid_cate_buy_time_daydiff']
# return df_user_order_features

######################################### get_user_loan_features() ###################################
# user loan features
# group by uid
dataset_loan = pd.read_csv(raw_dataset_path+'t_loan.csv')

dataset_loan_set1 = dataset_loan[(dataset_loan['loan_time']>'2016-08-03') & (dataset_loan['loan_time']<'2016-09-02')]

user_loan_features = {}

df_tmp = dataset_loan_set1[['uid','loan_time']]
user_loan_features['uid_loan_cnt'] = df_tmp.groupby(['uid']).count().iloc[:,0]

df_tmp = dataset_loan_set1[['uid','loan_time']]
user_loan_features['uid_loan_time_min'] = df_tmp.groupby(['uid']).agg('min')['loan_time']

df_tmp = dataset_loan_set1[['uid','loan_amount']]
user_loan_features['uid_loan_amount_min'] = df_tmp.groupby(['uid']).agg('min')['loan_amount']

df_tmp = dataset_loan_set1[['uid','plannum']]
user_loan_features['uid_plannum_min'] = df_tmp.groupby(['uid']).agg('min')['plannum']

df_tmp = dataset_loan_set1[['uid','loan_time']]
user_loan_features['uid_loan_time_max'] = df_tmp.groupby(['uid']).agg('max')['loan_time']

df_tmp = dataset_loan_set1[['uid','loan_amount']]
user_loan_features['uid_loan_amount_max'] = df_tmp.groupby(['uid']).agg('max')['loan_amount']

df_tmp = dataset_loan_set1[['uid','plannum']]
user_loan_features['uid_plannum_max'] = df_tmp.groupby(['uid']).agg('max')['plannum']

df_tmp = dataset_loan_set1[['uid','loan_amount']]
user_loan_features['uid_loan_amount_sum'] = df_tmp.groupby(['uid']).sum()['loan_amount']

df_tmp = dataset_loan_set1[['uid','plannum']]
user_loan_features['uid_plannum_sum'] = df_tmp.groupby(['uid']).sum()['plannum']

df_tmp = dataset_loan_set1[['uid','loan_amount']]
user_loan_features['uid_loan_amount_avg'] = df_tmp.groupby(['uid']).agg('mean')['loan_amount']

df_tmp = dataset_loan_set1[['uid','plannum']]
user_loan_features['uid_plannum_avg'] = df_tmp.groupby(['uid']).agg('mean')['plannum']

user_loan_features_on_uid = pd.DataFrame(user_loan_features)
# user loan features on plannum
# group by uid,plannum
user_loan_features = {}

df_tmp = dataset_loan_set1[['uid','plannum','loan_time']]
user_loan_features['uid_plannum_loan_cnt'] = df_tmp.groupby(['uid','plannum']).count().iloc[:,0]

df_tmp = dataset_loan_set1[['uid','plannum','loan_time']]
user_loan_features['uid_plannum_loan_time_min'] = df_tmp.groupby(['uid','plannum']).agg('min')['loan_time']

df_tmp = dataset_loan_set1[['uid','plannum','loan_amount']]
user_loan_features['uid_plannum_loan_amount_min'] = df_tmp.groupby(['uid','plannum']).agg('min')['loan_amount']

df_tmp = dataset_loan_set1[['uid','plannum','loan_time']]
user_loan_features['uid_plannum_loan_time_max'] = df_tmp.groupby(['uid','plannum']).agg('max')['loan_time']

df_tmp = dataset_loan_set1[['uid','plannum','loan_amount']]
user_loan_features['uid_plannum_loan_amount_max'] = df_tmp.groupby(['uid','plannum']).agg('max')['loan_amount']

df_tmp = dataset_loan_set1[['uid','plannum','loan_amount']]
user_loan_features['uid_plannum_loan_amount_sum'] = df_tmp.groupby(['uid','plannum']).sum()['loan_amount']

df_tmp = dataset_loan_set1[['uid','plannum','loan_amount']]
user_loan_features['uid_plannum_loan_amount_avg'] = df_tmp.groupby(['uid','plannum']).agg('mean')['loan_amount']

user_loan_features_on_uid_plannum = pd.DataFrame(user_loan_features)
user_loan_features_on_uid_plannum = user_loan_features_on_uid_plannum.unstack(level=-1)
# user_loan_features['uid_plannum_loan_time_daydiff']
# user_loan_features['uid_loan_time_daydiff']

######################################### get_user_click_features() ###################################
# user click features
# group by uid
dataset_click = pd.read_csv(raw_dataset_path+'t_click.csv')

dataset_click_set1 = dataset_click[(dataset_click['click_time']>'2016-08-03') & (dataset_click['click_time']<'2016-09-02')]

user_click_features = {}
# dataset_click_set1.groupby(['uid']).agg({'pid': np.max, 'pid': np.sum,'cnt': np.count})
# t4 = merchant1[(merchant1.date!='null')&(merchant1.coupon_id!='null')][['merchant_id','distance']]

df_tmp = dataset_click_set1[['uid','pid']]
user_click_features['user_click_cnt'] = df_tmp.groupby(['uid']).count().iloc[:,0]

df_tmp = dataset_click_set1[['uid','param']]
user_click_features['user_click_param_cnt'] = df_tmp.groupby(['uid']).sum()['param']

df_tmp = dataset_click_set1[['uid','click_time']]
user_click_features['user_click_time_min'] = df_tmp.groupby(['uid']).agg('min')['click_time']

df_tmp = dataset_click_set1[['uid','click_time']]
user_click_features['user_click_time_max'] = df_tmp.groupby(['uid']).agg('max')['click_time']

df_tmp = dataset_click_set1[['uid','pid']]
user_click_features['user_click_pid_cnt'] = df_tmp.groupby(['uid']).agg({'pid': pd.Series.nunique}).pid

df_tmp = dataset_click_set1[['uid','param']]
user_click_features['user_click_param_cnt'] = df_tmp.groupby(['uid']).agg({'param': pd.Series.nunique}).param

user_click_features_on_uid = pd.DataFrame(user_click_features)
# user click features on pid
# group by uid,pid
user_click_features = {}
df_tmp = dataset_click_set1[['uid','pid','click_time']]
user_click_features['user_click_pid_cnt'] = df_tmp.groupby(['uid','pid']).count().iloc[:,0]

df_tmp = dataset_click_set1[['uid','pid','click_time']]
user_click_features['uid_pid_click_time_min'] = df_tmp.groupby(['uid','pid']).agg('min')['click_time']

df_tmp = dataset_click_set1[['uid','pid','click_time']]
user_click_features['uid_pid_click_time_max'] = df_tmp.groupby(['uid','pid']).agg('max')['click_time']

df_tmp = dataset_click_set1[['uid','pid','param']]
user_click_features['user_click_pid_param_cnt'] = df_tmp.groupby(['uid','pid']).agg({'param': pd.Series.nunique}).param

user_click_features_on_uid_pid = pd.DataFrame(user_click_features)
user_click_features_on_uid_pid = user_click_features_on_uid_pid.unstack(level=-1)
# user click features on param
# group by uid,param
user_click_features = {}
df_tmp = dataset_click_set1[['uid','param','click_time']]
user_click_features['user_click_pid_cnt'] = df_tmp.groupby(['uid','param']).count().iloc[:,0]

df_tmp = dataset_click_set1[['uid','param','click_time']]
user_click_features['uid_pid_click_time_min'] = df_tmp.groupby(['uid','param']).agg('min')['click_time']

df_tmp = dataset_click_set1[['uid','param','click_time']]
user_click_features['uid_pid_click_time_max'] = df_tmp.groupby(['uid','param']).agg('max')['click_time']

df_tmp = dataset_click_set1[['uid','param','pid']]
user_click_features['user_click_param_pid_cnt'] = df_tmp.groupby(['uid','param']).agg({'pid': pd.Series.nunique}).pid

user_click_features_on_uid_param = pd.DataFrame(user_click_features)
user_click_features_on_uid_param = user_click_features_on_uid_param.unstack(level=-1)
# user click features on pid,param
# group by uid,pid,param
user_click_features = {}
df_tmp = dataset_click_set1[['uid','pid','param','click_time']]
user_click_features['user_pid_param_click_cnt'] = df_tmp.groupby(['uid','pid','param']).count().iloc[:,0]

df_tmp = dataset_click_set1[['uid','pid','param','click_time']]
user_click_features['uid_param_pid_click_time_min'] = df_tmp.groupby(['uid','pid','param']).agg('min')['click_time']

df_tmp = dataset_click_set1[['uid','pid','param','click_time']]
user_click_features['uid_param_pid_click_time_max'] = df_tmp.groupby(['uid','pid','param']).agg('max')['click_time']

user_click_features_on_uid_pid_param = pd.DataFrame(user_click_features)
user_click_features_on_uid_pid_param = user_click_features_on_uid_pid_param.unstack(level=-1)

# user_click_features['uid_param_pid_click_time_daydiff'] = 
# user_click_features['uid_pid_click_time_daydiff'] = 
# user_click_features['uid_click_time_daydiff'] = df_tmp.groupby(['uid']).agg('max')['click_time']
# return df_user_click_features

######################################### get_sex_features() ###################################
# sex features
# group by sex
dataset_sex = pd.read_csv(raw_dataset_path+'t_user.csv')

sex_features = {}

df_tmp = dataset_sex[['sex','active_date']]
sex_features['sex_cnt'] = df_tmp.groupby(['sex']).count().iloc[:,0]

df_tmp = dataset_sex[['sex','active_date']]
sex_features['sex_active_date_min'] = df_tmp.groupby(['sex']).agg('min')['active_date']

df_tmp = dataset_sex[['sex','limit']]
sex_features['sex_limit_min'] = df_tmp.groupby(['sex']).agg('min')['limit']

df_tmp = dataset_sex[['sex','active_date']]
sex_features['sex_active_date_max'] = df_tmp.groupby(['sex']).agg('max')['active_date']

df_tmp = dataset_sex[['sex','limit']]
sex_features['sex_limit_max'] = df_tmp.groupby(['sex']).agg('max')['limit']

df_tmp = dataset_sex[['sex','limit']]
sex_features['sex_limit_avg'] = df_tmp.groupby(['sex']).agg('mean')['limit']

df_tmp = dataset_sex[['sex','limit']]
sex_features['sex_limit_sum'] = df_tmp.groupby(['sex']).sum()['limit']

# sex features on order
# FROM `t_order` t1
# LEFT JOIN t_user t2 ON t1.`uid`=t2.`uid`
# group by sex 
dataset_sex = pd.read_csv(raw_dataset_path+'t_user.csv')
dataset_order = pd.read_csv(raw_dataset_path+'t_order.csv')
dataset_order_set1 = dataset_order[(dataset_order['buy_time']>'2016-08-03') & (dataset_order['buy_time']<'2016-09-02')]

dataset_sex_order_set1 = pd.merge(dataset_sex,dataset_order_set1,on=['uid'],how='left')

df_tmp = dataset_sex_order_set1[['sex','buy_time']]
sex_features['sex_order_cnt'] = df_tmp.groupby(['sex']).count().iloc[:,0]

df_tmp = dataset_sex_order_set1[['sex','price']]
sex_features['sex_price_sum'] = df_tmp.groupby(['sex']).sum()['price']

df_tmp = dataset_sex_order_set1[['sex','discount']]
sex_features['sex_discount_sum'] = df_tmp.groupby(['sex']).sum()['discount']

df_tmp = dataset_sex_order_set1[['sex','qty']]
sex_features['sex_qty_sum'] = df_tmp.groupby(['sex']).sum()['qty']

df_tmp = dataset_sex_order_set1[['sex','qty']]
sex_features['sex_qty_min'] = df_tmp.groupby(['sex']).agg('min')['qty']

df_tmp = dataset_sex_order_set1[['sex','buy_time']]
sex_features['sex_buy_time_min'] = df_tmp.groupby(['sex']).agg('min')['buy_time']

df_tmp = dataset_sex_order_set1[['sex','qty']]
sex_features['sex_qty_max'] = df_tmp.groupby(['sex']).agg('max')['qty']

df_tmp = dataset_sex_order_set1[['sex','buy_time']]
sex_features['sex_buy_time_max'] = df_tmp.groupby(['sex']).agg('max')['buy_time']

df_tmp = dataset_sex_order_set1[['sex','qty']]
sex_features['sex_qty_avg'] = df_tmp.groupby(['sex']).agg('mean')['qty']

df_tmp = dataset_sex_order_set1[['sex','price']]
sex_features['sex_price_avg'] = df_tmp.groupby(['sex']).agg('mean')['price']

df_tmp = dataset_sex_order_set1[['sex','discount']]
sex_features['sex_discount_avg'] = df_tmp.groupby(['sex']).agg('mean')['discount']

df_tmp = dataset_sex_order_set1[['sex','qty']]
sex_features['sex_qty_cnt'] = df_tmp.groupby(['sex']).agg({'qty': pd.Series.nunique}).qty

df_tmp = dataset_sex_order_set1[['sex','cate_id']]
sex_features['sex_cate_cnt'] = df_tmp.groupby(['sex']).agg({'cate_id': pd.Series.nunique}).cate_id

# sex features on loan
# FROM `t_loan` t1
# LEFT JOIN t_user t2 ON t1.`uid`=t2.`uid`
# group by sex
dataset_sex = pd.read_csv(raw_dataset_path+'t_user.csv')
dataset_loan = pd.read_csv(raw_dataset_path+'t_loan.csv')
dataset_loan_set1 = dataset_loan[(dataset_loan['loan_time']>'2016-08-03') & (dataset_loan['loan_time']<'2016-09-02')]

dataset_sex_loan_set1 = pd.merge(dataset_sex,dataset_loan_set1,on=['uid'],how='left')

df_tmp = dataset_sex_loan_set1[['sex','loan_time']]
sex_features['sex_loan_time_min'] = df_tmp.groupby(['sex']).agg('min')['loan_time']
sex_features['sex_loan_time_max'] = df_tmp.groupby(['sex']).agg('max')['loan_time']

df_tmp = dataset_sex_loan_set1[['sex','loan_amount']]
sex_features['sex_loan_amount_min'] = df_tmp.groupby(['sex']).agg('min')['loan_amount']
sex_features['sex_loan_amount_max'] = df_tmp.groupby(['sex']).agg('max')['loan_amount']
sex_features['sex_loan_amount_avg'] = df_tmp.groupby(['sex']).agg('mean')['loan_amount']
sex_features['sex_loan_amount_sum'] = df_tmp.groupby(['sex']).sum()['loan_amount']

df_tmp = dataset_sex_loan_set1[['sex','plannum']]
sex_features['sex_plannum_min'] = df_tmp.groupby(['sex']).agg('min')['plannum']
sex_features['sex_plannum_max'] = df_tmp.groupby(['sex']).agg('max')['plannum']
sex_features['sex_plannum_avg'] = df_tmp.groupby(['sex']).agg('mean')['plannum']
sex_features['sex_plannum_sum'] = df_tmp.groupby(['sex']).sum()['plannum']

sex_features_df = pd.DataFrame(sex_features)

# sex and plannum features on loan
# FROM `t_loan` t1
# LEFT JOIN t_user t2 ON t1.`uid`=t2.`uid`
# group by sex,plannum
sex_features = {}
df_tmp = dataset_sex_loan_set1[['sex','plannum','loan_amount']]
sex_features['sex_plannum_loan_amount_min'] = df_tmp.groupby(['sex','plannum']).agg('min')['loan_amount']
sex_features['sex_plannum_loan_amount_max'] = df_tmp.groupby(['sex','plannum']).agg('max')['loan_amount']
sex_features['sex_plannum_loan_amount_avg'] = df_tmp.groupby(['sex','plannum']).agg('mean')['loan_amount']
sex_features['sex_plannum_loan_amount_sum'] = df_tmp.groupby(['sex','plannum']).sum()['loan_amount']

df_tmp = dataset_sex_loan_set1[['sex','plannum','loan_time']]
sex_features['sex_plannum_loan_time_min'] = df_tmp.groupby(['sex','plannum']).agg('min')['loan_time']
sex_features['sex_plannum_loan_time_max'] = df_tmp.groupby(['sex','plannum']).agg('max')['loan_time']

sex_features_df_on_plannum = pd.DataFrame(sex_features)
sex_features_df_on_plannum = sex_features_df_on_plannum.unstack(level=-1)

"""
# sex and cate_id features on order
# FROM `t_order` t1
# LEFT JOIN t_user t2 ON t1.`uid`=t2.`uid`
# group by sex
# sex_features = {}
# df_tmp = dataset_sex_order_set1[['sex','cate_id','buy_time']]
# sex_features['uid_cate_order_cnt'] = df_tmp.groupby(['sex','cate_id']).count().iloc[:,0]
# sex_features['uid_cate_buy_time_min'] = df_tmp.groupby(['sex','cate_id']).agg('min')['buy_time']
# sex_features['uid_cate_buy_time_max'] = df_tmp.groupby(['sex','cate_id']).agg('max')['buy_time']
#
# df_tmp = dataset_sex_order_set1[['sex','cate_id','price']]
# sex_features['uid_cate_price_sum'] = df_tmp.groupby(['sex','cate_id']).sum()['price']
# sex_features['uid_cate_price_avg'] = df_tmp.groupby(['sex','cate_id']).agg('mean')['price']
#
#
# df_tmp = dataset_sex_order_set1[['sex','cate_id','discount']]
# sex_features['uid_cate_discount_sum'] = df_tmp.groupby(['sex','cate_id']).sum()['discount']
# sex_features['uid_cate_discount_avg'] = df_tmp.groupby(['sex','cate_id']).agg('mean')['discount']
#
# df_tmp = dataset_sex_order_set1[['sex','cate_id','qty']]
# sex_features['uid_cate_qty_sum'] = df_tmp.groupby(['sex','cate_id']).sum()['qty']
# sex_features['uid_cate_qty_min'] = df_tmp.groupby(['sex','cate_id']).agg('min')['qty']
# sex_features['uid_cate_qty_max'] = df_tmp.groupby(['sex','cate_id']).agg('max')['qty']
# sex_features['uid_cate_qty_avg'] = df_tmp.groupby(['sex','cate_id']).agg('mean')['qty']
# sex_features['uid_cate_qty_cnt'] = df_tmp.groupby(['sex','cate_id']).agg({'qty': pd.Series.nunique}).qty
#
# sex_features_df_on_cate = pd.DataFrame(sex_features)
# sex_features_df_on_cate = sex_features_df_on_cate.unstack(level=-1)

# sex features on click
# FROM `t_click` t1
# LEFT JOIN t_user t2 ON t1.`uid`=t2.`uid`
# GROUP BY t2.sex
sex_features = {}
dataset_sex = pd.read_csv(raw_dataset_path+'t_user.csv')
dataset_click = pd.read_csv(raw_dataset_path+'t_click.csv')
dataset_click_set1 = dataset_click[(dataset_click['buy_time']>'2016-08-03') & (dataset_click['buy_time']<'2016-09-02')]

dataset_sex_click_set1 = pd.merge(dataset_sex,dataset_click_set1,on=['uid'],how='left')

df_tmp = dataset_sex_click_set1[['sex','click_time']]
sex_features['uid_click_time_min'] = df_tmp.groupby(['sex']).agg('min')['click_time']
sex_features['uid_click_time_max'] = df_tmp.groupby(['sex']).agg('max')['click_time']
sex_features['sex_click_cnt'] = df_tmp.groupby(['sex']).count().iloc[:,0]

# sex and pid features on click
df_tmp = dataset_sex_click_set1[['sex','pid','click_time','param']]
sex_features['uid_pid_click_time_min'] = df_tmp.groupby(['sex','pid']).agg('min')['click_time']
sex_features['uid_pid_click_time_max'] = df_tmp.groupby(['sex','pid']).agg('max')['click_time']
sex_features['sex_click_pid_cnt'] = df_tmp.groupby(['sex','pid']).count().iloc[:,0]
sex_features['sex_click_pid_param_cnt'] = df_tmp.groupby(['sex','pid']).agg({'param': pd.Series.nunique}).param

# sex and param features on click
df_tmp = dataset_sex_click_set1[['sex','param','click_time','pid']]
sex_features['uid_param_click_time_min'] = df_tmp.groupby(['sex','param']).agg('min')['click_time']
sex_features['uid_param_click_time_max'] = df_tmp.groupby(['sex','param']).agg('max')['click_time']
sex_features['sex_click_param_cnt'] = df_tmp.groupby(['sex','param']).count().iloc[:,0]
sex_features['sex_click_param_pid_cnt'] = df_tmp.groupby(['sex','param']).agg({'pid': pd.Series.nunique}).pid

# sex and param and pid features on click
df_tmp = dataset_sex_click_set1[['sex','param','pid','click_time']]
sex_features['uid_param_pid_click_time_min'] = df_tmp.groupby(['sex','param','pid']).agg('min')['click_time']
sex_features['uid_param_pid_click_time_max'] = df_tmp.groupby(['sex','param','pid']).agg('max')['click_time']
sex_features['sex_param_pid_click_cnt'] = df_tmp.groupby(['sex','param','pid']).count().iloc[:,0]
"""

######################################### get_cate_features() ###################################
# cate features
# group by cate
dataset_cate = pd.read_csv(raw_dataset_path+'t_order.csv')
cate_features = {}

df_tmp = dataset_cate[['cate_id','buy_time']]
cate_features['cate_id_order_cnt'] = df_tmp.groupby(['cate_id']).count().iloc[:,0]
cate_features['cate_id_buy_time_min'] = df_tmp.groupby(['cate_id']).agg('min')['buy_time']
cate_features['cate_id_buy_time_max'] = df_tmp.groupby(['cate_id']).agg('max')['buy_time']

df_tmp = dataset_cate[['cate_id','price']]
cate_features['cate_id_price_sum'] = df_tmp.groupby(['cate_id']).sum()['price']
cate_features['cate_id_price_avg'] = df_tmp.groupby(['cate_id']).agg('mean')['price']

df_tmp = dataset_cate[['cate_id','discount']]
cate_features['cate_id_discount_sum'] = df_tmp.groupby(['cate_id']).sum()['discount']
cate_features['cate_id_discount_avg'] = df_tmp.groupby(['cate_id']).agg('mean')['discount']

df_tmp = dataset_cate[['cate_id','qty']]
cate_features['cate_id_qty_sum'] = df_tmp.groupby(['cate_id']).sum()['qty']
cate_features['cate_id_qty_avg'] = df_tmp.groupby(['cate_id']).agg('mean')['qty']
cate_features['cate_id_qty_min'] = df_tmp.groupby(['cate_id']).agg('min')['qty']
cate_features['cate_id_qty_max'] = df_tmp.groupby(['cate_id']).agg('max')['qty']

cate_features_df = pd.DataFrame(cate_features)

######################################### merge features ###################################
df_features = pd.merge(user_click_features_on_uid.reset_index()
                       ,user_order_features_on_uid_cate.reset_index()
                       ,on='uid',how='left')
df_features = pd.merge(df_features,user_loan_features_on_uid,on='uid',how='left')
