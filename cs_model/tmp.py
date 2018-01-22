# -*- coding:utf-8 -*-
__author__ = 'maomaochong'
import numpy as np
from sklearn.linear_model import LogisticRegression
import pandas as pd
import time
import pickle
import os
from woe.eval import  compute_ks

# 查看系数
# 加载模型文件
pkl_path = 'E:\\ScoreCard\\cs_model\\cs_m1_pos_model_daily\\gendata\\LogisticRegression_Model\\'
recent_path = max(os.listdir(pkl_path))
recent_path = pkl_path + recent_path
output = open(recent_path, 'rb')
ftrl = pickle.load(output)
output.close()
# 加载编码规则
enc_path = 'E:\\ScoreCard\\cs_model\\cs_m1_pos_model_daily\\gendata\\LogisticRegression_Model\\OneHotEncoder.pkl'
output = open(enc_path, 'rb')
enc = pickle.load(output)
output.close()