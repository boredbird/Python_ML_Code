# coding:utf-8
from sqlalchemy import *
from sqlalchemy.engine import create_engine
from sqlalchemy.schema import *
import pandas as pd
import time

# Presto 查询
engine = create_engine('presto://bqbpm2.bqjr.cn:8080/hive/nbr')
#查询表1
print 'reading  ',time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
df = pd.read_sql(sql=(r'select * from '+ 'nbr.phone_info_missing_20170810') , con=engine)
print 'done ',time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
# df.to_csv('E:\\phone_info_missing_20170810.csv', index=False)
#查询表2
print 'reading  ',time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
df1 = pd.read_sql(sql=(r'select * from '+ 'nbr.phonenum3_operator') , con=engine)
print 'done ',time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
# df1.to_csv('E:\\phonenum3_operator.csv', index=False)

#多个数据表写入同一个Excel文件
# 创建一个输出文件
writer = pd.ExcelWriter('E:\\out.xlsx')

df.to_excel(writer,'df',index=False)
df1.to_excel(writer,'df1',index=False)

writer.save()


