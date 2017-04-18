# -*- coding:utf-8 -*-
__author__ = 'chunhui.zhang'

import pandas as pd
import time

print 'START ',time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time()))

# file_name = ('JData_Action_201602','JData_Action_201603','JData_Action_201604','JData_Action_201603_extra',
             # 'JData_Comment','JData_Product','JData_User')

file_name = ('JData_Comment','JData_Product','JData_User')

adict = locals()

for name in file_name:
    print 'Reading: '+name
    try:
        print time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time()))
        #df = pd.read_csv('E:\\CIKM\\JData\\' + name +'.csv')
        adict['df_%s' % (name)] = pd.read_csv('E:\\CIKM\\JData\\' + name +'.csv')

    except Exception as e:
        print 'exception!'
