# -*- coding:utf-8 -*-
__author__ = 'maomaochong'
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns
import woe.config as config
sns.set_style("dark")   #设置背景色为黑色
from matplotlib.backends.backend_pdf import PdfPages
from pylab import *
mpl.rcParams['font.sans-serif'] = ['SimHei'] #指定默认字体
mpl.rcParams['axes.unicode_minus'] = False #解决保存图像是负号'-'显示为方块的问题

config_path = r'E:\Code\Python_ML_Code\cs_model\config\config_cs_model.csv'
dataset_path = r'E:\ScoreCard\cs_model\cs_m1_pos_model\raw_data\m1_rsx_cs_unify_model_features_201711.csv'
rst_path=r'E:\ScoreCard\cs_model\cs_m1_pos_model\gendata\WOE_Rule\cs_m1_pos_woe_rule_201710.pkl'
outfile_path=r'E:\ScoreCard\cs_model\cs_m1_pos_model\gendata\cs_m1_pos_woe_transed_rule_201710_features_201711.csv'

dataset = pd.read_csv(dataset_path)

def eavl_feature(dataset,var):
    dataset.loc[dataset[var].isnull(), (var)] = -1

    sub_set = dataset[dataset.target >=0][[var,'target']]
    sub_set.sort_values(by = var,ascending=True, inplace=True)
    idx_list= np.linspace(0,sub_set.shape[0],100)
    sub_cnt_list = []
    positive_cnt_list = []
    positive_rate_list = []
    avg_values_list = []
    log_p_ratio_list = []
    for i in range(idx_list.__len__()-1):
        sub_set1 =  sub_set[int(idx_list[i]):int(idx_list[i+1])]
        sub_cnt_list.append(sub_set1.shape[0])
        positive_cnt_list.append(sum(sub_set1.target))
        positive_rate_list.append(sum(sub_set1.target)*1.0/sub_set1.shape[0])
        avg_values_list.append(sum(sub_set1[var])*1.0/sub_set1.shape[0])
        log_p_ratio_list.append(np.log(positive_rate_list[-1]*1.0/(1-positive_rate_list[-1])))

    exp_p_list = [1/(1+np.exp(-var)) for var in log_p_ratio_list]

    df_cor = pd.DataFrame()
    df_cor[var] = pd.Series(avg_values_list)
    df_cor['log_p_ratio_'+var] = pd.Series(log_p_ratio_list)
    df_cor['positive_rate_'+var] = pd.Series(positive_rate_list)
    df_cor['exp_p_list_'+var] = pd.Series(exp_p_list)

    return df_cor


cfg = config.config()
cfg.load_file(config_path)
cfg.global_bt = sum(dataset[dataset.target >=0].target)
cfg.global_gt = dataset[dataset.target >=0].shape[0] - cfg.global_bt

feature_cnt = cfg.bin_var_list.__len__()
page_cnt = feature_cnt
pp = PdfPages(r'E:\ScoreCard\cs_model\cs_m1_pos_model\eval\cs_m1_pos_features_eval.pdf')
for i in range(page_cnt):
    df_cor = eavl_feature(dataset,cfg.bin_var_list.values[i])
    plot_tmp = sns.lmplot(x="log_p_ratio"+var, y="positive_rate"+var, data=df_cor)
    pp.savefig()

pp.close()



#
# sns.pointplot(avg_values_list, positive_rate_list, alpha=0.2)
# sns.pointplot(avg_values_list, log_p_ratio_list, alpha=0.2)
#
# sns.pointplot(exp_p_list, positive_rate_list)

# sns.lmplot(x="avg_days", y="positive_rate", data=df_cor)
# sns.lmplot(x="avg_days", y="positive_rate", data=df_cor)
# sns.lmplot(x="exp_p_list", y="positive_rate", data=df_cor)
# sns.distplot(dataset['avg_days'][:10000], bins=100)
# sns.distplot(dataset['avg_days'], bins=100)

