# -*- coding: UTF-8 -*-
import MySQLdb
import sys
from sqlalchemy import create_engine
import pandas as pd
import time
#
# conn = MySQLdb.connect(host='10.83.4.61', user='root', passwd='123456', db='crawler', port=3306, use_unicode=True,
#                        charset="utf8")
# cur = conn.cursor()
#

reload(sys)
sys.setdefaultencoding('utf-8')
host = 'localhost'
port = 3306
db = 'dmr'
user = 'root'
password = '123456'

# engine = create_engine(str(r"mysql+mysqldb://%s:" + '%s' + "@%s/%s") % (user, password, host, db))
#
# try:
#     df = pd.read_sql(sql=r'select * from city', con=engine)
#     df.to_sql('test', con=engine, if_exists='append', index=False)
# except Exception as e:
#     print(e.message)


###pandas0.19.0已不再支持to_sql里面的flavor=’mysql’
engine = create_engine(str(r"mysql+mysqldb://%s:" + '%s' + "@%s/%s") % (user, password, host, db))
file_name = ('JData_Action_201602','JData_Action_201603','JData_Action_201604','JData_Action_201603_extra',
             'JData_Comment','JData_Product','JData_User')
for name in file_name:
    print 'running: '+name
    try:
        print time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time()))
        df = pd.read_csv('E:\\CIKM\\data_file\\JData\\' + name +'.csv')
        print time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
        df.to_sql(name, con=engine,flavor= None, if_exists='append', index=False,chunksize =2000000)
        print time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
    except Exception as e:
        print 'exception!'


# ValueError: database flavor mysql is not supported！！！！！
# conn = MySQLdb.connect(host="127.0.0.1", user="root", passwd="zch0302",
#                        db="dmr", port=3306, charset='utf8')
# print time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time()))
# df = pd.read_csv(r'E:\D
        # MR\data_file\JData\JData_User.csv')
# print time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
# df.to_sql(name='JData_User', con=conn, flavor='mysql', if_exists='append', index=False)
# print time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))