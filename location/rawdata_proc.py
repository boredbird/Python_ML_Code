# -*- coding:utf-8 -*-
import pandas as pd

raw_data_path = r'E:\output\rawdata\ccf_first_round_user_shop_behavior.csv'

dataset = pd.read_csv(raw_data_path)
dataset['wifi_infos'][0].split(";")

wifi_infos_split = dataset.loc[0,:]

dataset.loc[0,:]
col_name = 'wifi_infos'
var_else = [var for var in dataset.columns if var <> col_name]

df_raw = dataset

def wifi_infos_transf(df_raw,col_name):

    # 以；分割一行拆分成多行
    var_else = [var for var in df_raw.columns if var <> col_name]
    list_of_series = []
    list_of_splits = []
    for i in xrange(dataset.shape[0]):
        if i % 10000 == 0:
            print i

        df_raw_else = df_raw[var_else].loc[i, :]
        df_raw_wifi = df_raw['wifi_infos'][i].split(";")
        len_split = df_raw_wifi.__len__()

        list_of_series.extend([df_raw_else for j in xrange(len_split)])
        list_of_splits.extend(df_raw_wifi)

    return list_of_series,list_of_splits


list_of_series,list_of_splits = wifi_infos_transf(dataset,'wifi_infos')