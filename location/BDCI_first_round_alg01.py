# -*- coding:utf-8 -*-
"""
直接根据ccf_first_round_shop_info表提供的每个shop_id的经纬度，
evaluation_public数据集匹配shop_info经纬度，找到最近的shop_id作为结果提交
"""
__author__ = 'maomaochong'
import pandas as pd

"""
经纬度换算成球面距离 START
"""
from math import sin, asin, cos, radians, fabs, sqrt

EARTH_RADIUS=6367000.0            # 地球平均半径m

def hav(theta):
    s = sin(theta / 2)
    return s * s

def get_distance_hav(lat0, lng0, lat1, lng1):
    "用haversine公式计算球面两点间的距离。"
    # 经纬度转换成弧度
    lat0 = radians(lat0)
    lat1 = radians(lat1)
    lng0 = radians(lng0)
    lng1 = radians(lng1)

    dlng = fabs(lng0 - lng1)
    dlat = fabs(lat0 - lat1)
    h = hav(dlat) + cos(lat0) * cos(lat1) * hav(dlng)
    distance = 2 * EARTH_RADIUS * asin(sqrt(h))

    return distance

"""
经纬度换算成球面距离 END
"""

shop_info = pd.read_csv(r'E:\output\rawdata\ccf_first_round_shop_info.csv')
evalset = pd.read_csv(r'E:\output\rawdata\evaluation_public.csv')

#
# lon1,lat1 = (22.599578, 113.973129)
# lon2,lat2 = (22.6986848, 114.3311032)
# d2 = get_distance_hav(shop_info['longitude'][0],shop_info['latitude'][0],evalset['longitude'][0],evalset['latitude'][0])

fields_keep = ['row_id','user_id','mall_id','longitude_x','latitude_x','shop_id','longitude_y','latitude_y']
df_merged = pd.merge(evalset, shop_info, how='left', on=['mall_id'])[fields_keep]

# 此处内存已爆掉，改用sql完成
df_merged['distance'] = get_distance_hav(df_merged['longitude_x'],df_merged['latitude_x']
                                         ,df_merged['longitude_y'],df_merged['latitude_y'])


