# -*- coding:utf-8 -*-
"""
根据ccf_first_round_user_shop_behavior用户行为发生时的经纬度所对应的shop_id
evaluation_public数据集匹配user_shop_behavior经纬度，找到最近的K个shop_id，根据shop_id出现的次数排序最大的作为结果提交
"""
__author__ = 'maomaochong'
import pandas as pd
from util import  *

user_shop_behavior = pd.read_csv(r'E:\output\rawdata\ccf_first_round_user_shop_behavior.csv')
shop_info = pd.read_csv(r'E:\output\rawdata\ccf_first_round_shop_info.csv')
evalset = pd.read_csv(r'E:\output\rawdata\evaluation_public.csv')

