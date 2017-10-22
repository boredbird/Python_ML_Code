# -*- coding:utf-8 -*-
import pandas as pd
import time
raw_data_path = r'E:\output\rawdata\ccf_first_round_user_shop_behavior.csv'

dataset = pd.read_csv(raw_data_path)

col_name = 'wifi_infos'
var_else = [var for var in dataset.columns if var <> col_name]

def str_split(str):
    # 切分字符串
    return str.split(";")

dataset['wifi_splits'] = dataset['wifi_infos'].apply(str_split)


max([dataset['wifi_splits'][i].__len__() for i in xrange(dataset['wifi_splits'].__len__())])

def series_trans(dataset):
    # 一行转多行
    dataset_trans = pd.DataFrame({'user_id':dataset['user_id'],
                                   'shop_id':dataset['shop_id'],
                                   'time_stamp':dataset['time_stamp'],
                                   'longitude':dataset['longitude'],
                                   'latitude':dataset['latitude'],
                                   'wifi_infos':dataset['wifi_splits']})

    return  dataset_trans

# dataset_trans = dataset.apply(series_trans,axis=1)

print time.asctime( time.localtime(time.time()) )
dataset_trans = series_trans(dataset.loc[0,:])

# dataset.shape[0]
for i in xrange(1,dataset.shape[0]):
    if i % 10000 == 0:
        print i
        print time.asctime( time.localtime(time.time()) )

    dataset_trans = dataset_trans.append(series_trans(dataset.loc[i,:]))

dataset_transd = pd.concat(dataset_trans)

print dataset_transd.shape

def str_split2(str):
    # 切分字符串
    return str.split("|")

dataset_transd['bssid'] = dataset_transd['wifi_infos'].apply(str_split2)[0]
dataset_transd['signal'] = dataset_transd['wifi_infos'].apply(str_split2)[1]
dataset_transd['flag'] = dataset_transd['wifi_infos'].apply(str_split2)[2]



# df.loc[i]={'a':1,'b':2}

