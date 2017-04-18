import MySQLdb

# try:
#     conn = MySQLdb.connect(host='192.168.2.100', user='root', passwd='123456', db='crawler', port=3306)
#     cur = conn.cursor()
#     cur.execute('select * from crawler.stock_pd_tech_vane')
#     cur.close()
#     conn.close()
# except MySQLdb.Error, e:
#     print "Mysql Error %d: %s" % (e.args[0], e.args[1])


#!/usr/bin/python
#
# import time
#
# print (time.strftime("%Y-%m-%d"))

#cur.execute("insert into crawler.stock_pd_tech_vane values('2','Tom','3 year 2 class','9')")

conn = MySQLdb.connect(host='localhost', user='root', passwd='123456', db='crawler', port=3306)
cur = conn.cursor()
count = cur.execute('select stock_no from crawler.stock_pd_price')
print count
result = cur.fetchmany(5)
for i in result :
    print i[2]
    print type(i[2])

cur.close()
conn.close()


