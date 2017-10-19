# -*- coding:utf-8 -*-
__author__ = 'chunhui.zhang'

file_object = open(r'/home/zch/hqlscripts/summary0717.sql')
try:
     all_the_text = file_object.read( )
finally:
     file_object.close( )

sql_list = all_the_text.split(';')


#使用pyhs2连接hive
import pyhs2
conn=pyhs2.connect(host='bqbpm2.bqjr.cn',port=10000, authMechanism="PLAIN",user='chunhui.zhang', password='thinkabout1',database='nbr')

def sql_executor(conn,sql):
    conn = conn
    sql = sql
    cursor = conn.cursor()
    cursor.execute(sql)
    cursor.close()
    return 1

for i in range(0,len(sql_list)):
    print sql_list[i]
    sql_executor(conn,sql_list[i])


# conn.close()
