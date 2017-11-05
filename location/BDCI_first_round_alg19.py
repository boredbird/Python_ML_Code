# -*- coding: utf-8 -*-
"""
xgb
有效wifi
wifi的信号强度
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
log_file = open(r'E:\output\gendata\BDCI_first_round_alg19_submit01.log', "w")
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

wifi_time = defaultdict(lambda: [])
for line in user_shop_behavior.values:
    for wifi in line[5].split(';'):
        wifi_time[wifi.split('|')[0]].append(line[2])

a = [[k,min(v),max(v)] for k,v in wifi_time.items()] # 399679
b = pd.DataFrame(a,columns=['wifi_id','min_time','max_time'])
day_diff = [(parser.parse(var[2])-parser.parse(var[1])).days for var in b.values]
b['day_diff'] = day_diff

valid_wifi = b[b['day_diff']>0] # 139542
valid_wifi.index = valid_wifi['wifi_id']


"""
alg10:start
"""
import time
# 训练规则
wifi_to_shops = defaultdict(lambda : defaultdict(lambda :0))
print time.asctime(time.localtime(time.time()))
for line in user_shop_behavior.values:
    try:
        wifi = sorted([wifi.split('|') for wifi in line[5].split(';') if wifi.split('|')[0] in  valid_wifi.index
                       ],key=lambda x:int(x[1]),reverse=True)[0]
        wifi_to_shops[wifi[0]][line[1]] = wifi_to_shops[wifi[0]][line[1]] + 1
    except:
        continue

print time.asctime(time.localtime(time.time()))

def get_wifi_to_shop(wifi_infos):
    try:
        wifi = sorted([wifi.split('|') for wifi in wifi_infos.split(';')],key=lambda x:int(x[1]),reverse=True)[0]
        counter = defaultdict(lambda : 0)
        for k,v in wifi_to_shops[wifi[0]].items():
            counter[k] += v
        pred_one = sorted(counter.items(),key=lambda x:x[1],reverse=True)[0][0]
    except:
        pred_one = 's_666'
    return pred_one

"""
alg10:end
"""

dataset =pd.concat([user_shop_behavior,evalset])
mall_list = list(set(list(shop_info.mall_id)))
result=pd.DataFrame()
for mall in mall_list:
    print '\n'
    print '[PROC]', '=' * (40 - len(mall)/2), mall,'=' * (40 - len(mall)/2)
    print '\n'
    wifi_list = []
    train_segment = user_shop_behavior.loc[mall]
    train_segment = train_segment.reset_index(drop=True)
    test_segment = evalset.loc[mall].reset_index(drop=True)
    segment = pd.concat([train_segment, test_segment]).reset_index(drop=True)

    for line in segment.values:
        wifi = {}
        for var in line[7].split(';'):
            if var.split('|')[0] in  valid_wifi.index:
                wifi[var.split('|')[0]] = int(var.split('|')[1])
        wifi['wifi_to_shop'] = get_wifi_to_shop(line[7])
        wifi_list.append(wifi)

    train_ext = pd.concat([segment,pd.DataFrame(wifi_list)], axis=1)
    # 增加时间特征
    train_ext.time_stamp = [int(var[11:13]) for var in train_ext.time_stamp]
    # 增加alg10规则预测结果作为特征输入
    wifi_to_shop_lbl = preprocessing.LabelEncoder()
    wifi_to_shop_lbl.fit(list(train_ext['wifi_to_shop'].values))
    train_ext['wifi_to_shop_label'] = wifi_to_shop_lbl.transform(list(train_ext['wifi_to_shop'].values))

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
        'nthread': 8
    }

    feature = [x for x in train_ext.columns if x not in ['user_id', 'label', 'shop_id', 'mall_id', 'wifi_infos', 'wifi_to_shop']] # 'time_stamp','wifi_to_shop_label'
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
    result.to_csv(r'E:\output\submit\BDCI_first_round_alg19_submit01.csv', index=False)

######################################################
log_file.close()
# restore the output to initial pattern
sys.stdout = stdout_backup

print "Now this will be presented on screen"

# score: 0.