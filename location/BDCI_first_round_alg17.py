# -*- coding: utf-8 -*-
"""
xgb
取消有效wifi的限制
wifi的信号强度，改为 wifi信号强度的rank
时间特征
增加shop_id特征：取值0或者1
根据经纬度规则判定的shop_id为1，其它为0
根据wifi信号强度最强规则判定的shop_id为1，其它为0，alg10的结果作为输入
"""
from sklearn import  preprocessing
import pandas as pd
from collections import defaultdict
from dateutil import parser
import sys
import xgboost as xgb

# make a copy of original stdout route
stdout_backup = sys.stdout
# define the log file that receives your log info
log_file = open(r'E:\output\gendata\BDCI_first_round_alg17_submit01.log', "w")
# redirect print output to log file
sys.stdout = log_file

print "Now all print info will be written to message.log"
# any command line that you will execute
######################################################
user_shop_behavior = pd.read_csv(r'E:\output\rawdata\ccf_first_round_user_shop_behavior.csv')
shop_info = pd.read_csv(r'E:\output\rawdata\ccf_first_round_shop_info.csv')
evalset = pd.read_csv(r'E:\output\rawdata\evaluation_public.csv')

nearest = []
shop_info.index = shop_info['shop_id']

user_shop_behavior['mall_id'] = shop_info.loc[user_shop_behavior['shop_id'] ,]['mall_id'].tolist()
user_shop_behavior.index = user_shop_behavior['mall_id']
evalset.index = evalset['mall_id']

dataset =pd.concat([user_shop_behavior,evalset])
mall_list = list(set(list(shop_info.mall_id)))
result=pd.DataFrame()
for mall in mall_list:
    wifi_list = []
    train_segment = user_shop_behavior.loc[mall]
    train_segment = train_segment.reset_index(drop=True)
    test_segment = evalset.loc[mall].reset_index(drop=True)
    segment = pd.concat([train_segment, test_segment]).reset_index(drop=True)

    for line in segment.values:
        wifi = {}
        for var in line[7].split(';'):
            wifi[var.split('|')[0]] = int(var.split('|')[1])
        wifi_list.append(wifi)

    train_ext = pd.concat([segment,pd.DataFrame(wifi_list)], axis=1)

    df_train = train_ext[train_ext.shop_id.notnull()]
    df_test = train_ext[train_ext.shop_id.isnull()]

    lbl = preprocessing.LabelEncoder()
    lbl.fit(list(df_train['shop_id'].values))
    df_train['label'] = lbl.transform(list(df_train['shop_id'].values))

    num_class = df_train['label'].max() + 1
    params = {
        'objective': 'multi:softmax',
        'eta': 0.1,
        'max_depth': 9,
        'eval_metric': 'merror',
        'seed': 0,
        'missing': -999,
        'num_class': num_class,
        'silent': 1,
        'nthread': 7
    }

    feature = [x for x in train_ext.columns if x not in ['user_id', 'label', 'shop_id', 'time_stamp', 'mall_id', 'wifi_infos']]
    xgbtrain = xgb.DMatrix(df_train[feature], df_train['label'])
    xgbtest = xgb.DMatrix(df_test[feature])
    watchlist = [(xgbtrain, 'train'), (xgbtrain, 'test')]
    num_rounds = 100

    model = xgb.train(params, xgbtrain, num_rounds, watchlist, early_stopping_rounds=30)

    df_test['label'] = model.predict(xgbtest)
    df_test['shop_id'] = df_test['label'].apply(lambda x: lbl.inverse_transform(int(x)))

    r = df_test[['row_id', 'shop_id']]
    result = pd.concat([result, r])
    result['row_id'] = result['row_id'].astype('int')
    result.to_csv(r'E:\output\submit\BDCI_first_round_alg17_submit01.csv', index=False)

######################################################
log_file.close()
# restore the output to initial pattern
sys.stdout = stdout_backup

print "Now this will be presented on screen"