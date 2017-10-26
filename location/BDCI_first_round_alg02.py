# -*- coding:utf-8 -*-
"""
根据ccf_first_round_user_shop_behavior用户行为发生时的经纬度所对应的shop_id
evaluation_public数据集匹配user_shop_behavior经纬度，找到最近的shop_id作为结果提交
"""
__author__ = 'maomaochong'
import pandas as pd
from location.util import  *
import time
from multiprocessing import Pool

"""
预测
"""
# 写法一：
# print time.asctime( time.localtime(time.time()) )
# for line in evalset.values:
#     shop_distance = [[get_distance_hav(line[4],line[5],shop[0],shop[1]),shop[2]] for shop in user_shop_behavior.loc[line[2],][['longitude','latitude','shop_id']].values]
#     nearest.append(sorted(shop_distance)[0][1])
#     if nearest.__len__() == 1000:
#         print time.asctime(time.localtime(time.time()))
#         break
"""
Wed Oct 25 15:02:45 2017
Wed Oct 25 15:03:46 2017
"""


# 写法二：
# print time.asctime( time.localtime(time.time()) )
# for line in evalset.values:
#     tmp_shops = user_shop_behavior[user_shop_behavior['mall_id'] == line[2]][['longitude','latitude','shop_id']].values
#     shop_distance = [[get_distance_hav(line[4],line[5],shop[0],shop[1]),shop[2]] for shop in tmp_shops]
#     nearest.append(sorted(shop_distance)[0][1])
#     if nearest.__len__() == 1000:
#         print time.asctime(time.localtime(time.time()))
#         break
"""
Wed Oct 25 14:58:35 2017
Wed Oct 25 15:00:37 2017
"""

# 写法三：

user_shop_behavior = pd.read_csv(r'E:\output\rawdata\ccf_first_round_user_shop_behavior.csv')
shop_info = pd.read_csv(r'E:\output\rawdata\ccf_first_round_shop_info.csv')
evalset = pd.read_csv(r'E:\output\rawdata\evaluation_public.csv')

nearest = []
shop_info.index = shop_info['shop_id']

user_shop_behavior['mall_id'] = shop_info.loc[user_shop_behavior['shop_id'] ,]['mall_id'].tolist()
user_shop_behavior.index = user_shop_behavior['mall_id']

counter = 0
def get_nearest(line):
    global counter
    counter = counter+1
    if counter%1000 == 0:
        print counter
        print time.asctime(time.localtime(time.time()))

    # tmp_shops = user_shop_behavior[user_shop_behavior['mall_id'] == line[2]][['longitude', 'latitude', 'shop_id']].values
    tmp_shops = user_shop_behavior.loc[line[2],][['longitude', 'latitude', 'shop_id']].values
    shop_distance = [[get_distance_hav(line[4], line[5], shop[0], shop[1]), shop[2]] for shop in tmp_shops]
    nearest.append(sorted(shop_distance)[0][1])

    return {'row_id':line[0],'shop_id':sorted(shop_distance)[0][1]}

if __name__ == "__main__":

    dataset = evalset.values
    pool = Pool()
    result = pool.map(get_nearest,dataset)
    pool.close()
    pool.join()

    result = pd.DataFrame(result)
    result.to_csv(r'E:\output\submit\BDCI_first_round_alg02_submit01.csv.csv',index=None)

#
# result = pd.DataFrame({'row_id':evalset.row_id,'shop_id':nearest})
# result.to_csv(r'E:\output\submit\BDCI_first_round_alg02_submit01.csv.csv',index=None)

# score:0.6329

"""
线下验证
"""
# for line in user_shop_behavior.values:
#     shop_distance = [[get_distance_hav(line[4],line[5],shop[0],shop[1]),shop[2]] for shop in user_shop_behavior.loc[line[2],][['longitude','latitude','shop_id']].values]
#     nearest.append(sorted(shop_distance)[0][1])
#
