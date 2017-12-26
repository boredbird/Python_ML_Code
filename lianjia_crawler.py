# coding:utf-8
import requests
from bs4 import BeautifulSoup
import sys
import requests
import time
reload(sys)
sys.setdefaultencoding('utf-8')
today = time.strftime("%Y-%m-%d")
"""
https://sz.lianjia.com/ershoufang/rs/
https://sz.lianjia.com/ershoufang/pg2/
"""
url = requests.get('https://sz.lianjia.com/ershoufang/rs/')
url.encoding = 'gb2312'
# wbdata = requests.get(url).text
# 请求URL，获取其text文本
wbdata = url.text
soup = BeautifulSoup(wbdata, 'lxml')
a = soup.select(".tdc2")


lc = len(a[1].contents) - 1
print a[1].contents[lc]
province = a[1].contents[lc].split()[0]
if len(a[1].contents[lc].split()) > 1:
    city = a[1].contents[lc].split()[1]
else:
    city = province

