# -*- coding:utf-8 -*-
__author__ = 'chunhui.zhang'

import pandas as pd
import time


print time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time()))
df = pd.read_csv('E:\\CIKM\\data_file\\JData\\' + name +'.csv')
print time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
df.to_sql(name, con=engine,flavor= None, if_exists='append', index=False,chunksize =2000000)
print time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
