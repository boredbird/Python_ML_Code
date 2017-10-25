# -*- coding:utf-8 -*-
__author__ = 'maomaochong'

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