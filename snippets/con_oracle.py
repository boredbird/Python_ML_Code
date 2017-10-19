# coding:utf-8
import cx_Oracle                                          #引用模块cx_Oracle
conn=cx_Oracle.connect('zhangchunhui/RH5sKtHq970=@10.31.1.42/SAS')    #连接数据库
c=conn.cursor()                                           #获取cursor
x=c.execute('select sysdate from dual')                   #使用cursor进行各种操作
x.fetchone()
c.close()                                                 #关闭cursor
conn.close()
#关闭连接



#名字要写对，oracle的O字母是大写
import cx_Oracle
username="zhangchunhui"
userpwd="RH5sKtHq970="
host="10.31.1.42"
port=1521
dbname="SAS"
SID = orcl
dsn=cx_Oracle.makedsn(host, port, dbname)
connection=cx_Oracle.connect(username, userpwd, dsn)

##这个是可以的
connection = cx_Oracle.connect('zhangchunhui/RH5sKtHq970=@10.31.1.42/orcl')


cursor = connection.cursor()
sql = "select sysdate from dual"
cursor.execute(sql)
result = cursor.fetchall()
count = cursor.rowcount
print ("=====================" )
print ("Total:", count)
print ("=====================")
for row in result:
        print (row)
cursor.close()
connection.close()