# -*- coding:utf-8 -*-
"""
每个变量的IV值波动，用于监测特征的有效性
"""
__author__ = 'maomaochong'
import pickle
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from pylab import *
mpl.rcParams['font.sans-serif'] = ['SimHei'] #指定默认字体
mpl.rcParams['axes.unicode_minus'] = False #解决保存图像是负号'-'显示为方块的问题

rst_path_list = [
    r'E:\ScoreCard\cs_model\gendata\cs_m1_pos_woe_rule_201701.pkl',
    r'E:\ScoreCard\cs_model\gendata\cs_m1_pos_woe_rule_201702.pkl',
    r'E:\ScoreCard\cs_model\gendata\cs_m1_pos_woe_rule_201703.pkl',
    r'E:\ScoreCard\cs_model\gendata\cs_m1_pos_woe_rule_201704.pkl',
    r'E:\ScoreCard\cs_model\gendata\cs_m1_pos_woe_rule_201705.pkl',
    r'E:\ScoreCard\cs_model\gendata\cs_m1_pos_woe_rule_201706.pkl',
    r'E:\ScoreCard\cs_model\gendata\cs_m1_pos_woe_rule_201707.pkl',
    r'E:\ScoreCard\cs_model\gendata\cs_m1_pos_woe_rule_201708.pkl',
    r'E:\ScoreCard\cs_model\gendata\cs_m1_pos_woe_rule_201709.pkl'
]

# init monitor_iv_dict
monitor_iv_dict = {}
rst_path = rst_path_list[0]
output = open(rst_path, 'rb')
rst = pickle.load(output)
output.close()
for i in range(rst.__len__()):
    monitor_iv_dict[rst[i].var_name] = []

# extract iv value
for mon in range(rst_path_list.__len__()):
    rst_path = rst_path_list[mon]
    output = open(rst_path, 'rb')
    rst = pickle.load(output)
    output.close()

    for i in range(rst.__len__()):
        monitor_iv_dict[rst[i].var_name].append(rst[i].iv)

# output
pd.DataFrame(monitor_iv_dict).to_csv(r'E:\ScoreCard\cs_model\eval\cs_m1_pos_features_iv_monitor.csv')

# plot
monitor_iv_df = pd.DataFrame(monitor_iv_dict
                             ,index=['201701','201702','201703','201704','201705','201706','201707','201708','201709'])

feature_cnt = monitor_iv_df.columns.__len__()
page_cnt = feature_cnt/5 +1
pp = PdfPages(r'E:\ScoreCard\cs_model\eval\cs_m1_pos_features_iv_monitor.pdf')
for i in range(page_cnt):
    variables_show_list = monitor_iv_df.mean().sort_values(ascending=False)[5*i:5*(i+1)].index
    plot_tmp = monitor_iv_df[variables_show_list].plot(subplots=True
                                              ,title = "cs_m1_pos_features_iv_monitor feature index: "+str(5*i+1)+"-"+str(min(5*i+6,feature_cnt)))
    pp.savefig()

pp.close()





