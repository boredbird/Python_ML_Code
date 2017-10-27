# -*- coding:utf-8 -*-
"""
根据ccf_first_round_user_shop_behavior用户行为发生时的经纬度所对应的shop_id
evaluation_public数据集匹配user_shop_behavior经纬度，找到最近的K个shop_id，根据shop_id出现的次数排序最大的作为结果提交
"""
__author__ = 'maomaochong'
import pandas as pd
from location.util import  *
import time
from multiprocessing import Pool
from collections import Counter
import numpy as np

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

    tmp_shops = user_shop_behavior.loc[line[2],][['longitude', 'latitude', 'shop_id']].values
    shop_distance = np.array(sorted([[get_distance_hav(line[4], line[5], shop[0], shop[1]), shop[2]] for shop in tmp_shops])[:9])
    nearest = Counter(shop_distance[:,1]).most_common(1)[0][0]

    return {'row_id':line[0],'shop_id':nearest}

if __name__ == "__main__":

    dataset = evalset.values
    pool = Pool()
    result = pool.map(get_nearest,dataset)
    pool.close()
    pool.join()

    result = pd.DataFrame(result)
    result.to_csv(r'E:\output\submit\BDCI_first_round_alg03_submit_k9.csv.csv',index=None)


# k:3 score:0.6569
# k:5 score:0.6707
# k:7 score:0.6768
# k:9 score: