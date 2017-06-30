# -*- coding:utf-8 -*-
__author__ = 'chunhui.zhang'

import pandas as pd
import time
import traceback
import sys
from sqlalchemy import create_engine


def load_from_csv(path,file):
    adict = locals()

    try:
        for name in file:
            print 'reading: ' + name + time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
            adict['df_%s' % (name)] = pd.read_csv(path + name + '.csv')
            print 'done: ' + name + time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))

    except Exception, e:
        print 'str(Exception):\t', str(Exception)
        print 'str(e):\t\t', str(e)
        print 'repr(e):\t', repr(e)
        print 'e.message:\t', e.message
        # print 'traceback.print_exc():'
        # traceback.print_exc()
        # print 'traceback.format_exc():\n%s' % traceback.format_exc()

    return [adict['df_%s' % (name)] for name in file]


def load_into_csv(path,df,file_name=None):
    if file_name is None:
        file_name=df
    df.to_csv(path+file_name+'.csv', index=False)

    return  1


def load_into_mysql(df,tablename):
    reload(sys)
    # sys.setdefaultencoding('utf-8')
    host = 'localhost'
    port = 3306
    db = 'dmr'
    user = 'root'
    password = 'zch0302'

    engine = create_engine(str(r"mysql+mysqldb://%s:" + '%s' + "@%s/%s") % (user, password, host, db))

    try:
        print 'writing into mysql: ',tablename,' ',time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
        df.to_sql(tablename, con=engine, flavor=None, if_exists='append', index=False, chunksize=2000000)
        print 'done ',time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
    except Exception, e:
        print 'str(Exception):\t', str(Exception)
        print 'str(e):\t\t', str(e)
        print 'repr(e):\t', repr(e)
        print 'e.message:\t', e.message

    return  1

def load_from_mysql(tablename):
    reload(sys)
    # sys.setdefaultencoding('utf-8')
    host = 'localhost'
    port = 3306
    db = 'dmr'
    user = 'root'
    password = 'zch0302'

    engine = create_engine(str(r"mysql+mysqldb://%s:" + '%s' + "@%s/%s") % (user, password, host, db))

    try:
        print 'reading from mysql: ',tablename,' ',time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
        df = pd.read_sql(sql=(r'select * from '+ tablename) , con=engine)
        print 'done ',time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
    except Exception, e:
        print 'str(Exception):\t', str(Exception)
        print 'str(e):\t\t', str(e)
        print 'repr(e):\t', repr(e)
        print 'e.message:\t', e.message

    return  df