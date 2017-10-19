# -*- coding:utf-8 -*-
__author__ = 'chunhui.zhang'

from sqlalchemy import *
from sqlalchemy.engine import create_engine
from sqlalchemy.schema import *
import pandas as pd
import time

# Presto 增删改查 测试
engine = create_engine('presto://bqbpm2.bqjr.cn:8080/hive/nbr')
# Hive
# engine = create_engine('hive://localhost:10000/default')
# logs = Table('rpt_nbr_risk_d', MetaData(bind=engine), autoload=True)
# print select([func.count('*')], from_obj=logs).scalar()

# Presto 查询
print 'reading  ',time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
df = pd.read_sql(sql=(r'select * from '+ 'nbr.rpt_nbr_risk_d') , con=engine)
print 'done ',time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))

# Presto 插入
tablename = 'rpt_nbr_risk_d_bak'
print 'writing : ', tablename, ' ', time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
df.to_sql(tablename, con=engine, flavor=None, if_exists='append', index=False, chunksize=2000000)
print 'done ', time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))

# Presto 删除  Access Denied
tablename = 'rpt_nbr_risk_d_bak'
print 'writing : ', tablename, ' ', time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
rst = engine.execute("drop table rpt_nbr_risk_d_bak")
print rst
print 'done ', time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))


# Hive 增删改查 测试
engine = create_engine('hive://bqbpm2.bqjr.cn:8080/hive/nbr')
# Hive 查询
print 'reading  ',time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
df = pd.read_sql(sql=(r'select * from '+ 'nbr.rpt_nbr_risk_d') , con=engine)
print 'done ',time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))

# Hive 插入
tablename = 'rpt_nbr_risk_d_bak'
print 'writing : ', tablename, ' ', time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
df.to_sql(tablename, con=engine, flavor=None, if_exists='append', index=False, chunksize=2000000)
print 'done ', time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))

# Hive 删除  Access Denied
tablename = 'rpt_nbr_risk_d_bak'
print 'writing : ', tablename, ' ', time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
rst = engine.execute("drop table rpt_nbr_risk_d_bak")
print rst
print 'done ', time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))


#使用pyhs2连接hive
import pyhs2
# def database_connect():
# conn=pyhs2.connect(host='10.31.1.11',port=10000, authMechanism="PLAIN",user='root', password='flash!',database='portrait')
conn=pyhs2.connect(host='bqbpm2.bqjr.cn',port=10000, authMechanism="PLAIN",user='chunhui.zhang', password='thinkabout1',database='nbr')
cursor=conn.cursor()
# sql='select * from nbr.phone_info'
sql='create table  nbr.con_hive_test_create(id VARCHAR(20),var_name VARCHAR(40) )'
# sql1='select customerid,sex,age,marriage,unitkind,cellproperty,headship,selfmonthincome,familymonthincome,eduexperience from dx_cash_cus_info'
cursor.execute(sql)

sql='drop table  nbr.con_hive_test_create'
cursor.execute(sql)

result=cursor.fetch()
cursor.execute(sql1)
result1=cursor.fetch()
cursor.close()
conn.close()
# return result,result1


from pyhive import hive
cursor = hive.connect('bqbpm2.bqjr.cn').cursor()
cursor.execute('SELECT * FROM nbr.rpt_nbr_risk_d LIMIT 10')
print cursor.fetchone()
print cursor.fetchall()