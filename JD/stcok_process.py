# -*- coding: UTF-8 -*-
import MySQLdb
import sys
from sqlalchemy import create_engine
import pandas as pd
import time

reload(sys)
sys.setdefaultencoding('utf-8')
host = 'localhost'
port = 3306
db = 'crawler'
user = 'root'
password = '123456'

engine = create_engine(str(r"mysql+mysqldb://%s:" + '%s' + "@%s/%s") % (user, password, host, db) , encoding='utf8')
df = pd.read_sql(sql=r'select stock_no,stock_index,index_value,date from crawler.stock_pd_price', con=engine)

# df.head()
