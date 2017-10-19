# -*- coding:utf-8 -*-
__author__ = 'chunhui.zhang'
import sys
sys.path.append(r'E:\Code\ScoreCard')
from config_params import  *
import load_data as ld
import change_feature_dtype as cfd
import data_binning as bin
import calculate_iv as ci
import woe_transformation as wt
import numpy as np

rawdata_path = 'E:/ScoreCard/rawdata/'
raw_file_name = ['af_policy_temp', ]
train_file_name = 'af_policy_temp'
# model path
model_path = 'E:/Code/ScoreCard/'

config =ld.load_from_csv(model_path,['config',])
config = config[0]

variable_type = config[['var_name','var_dtype']]
variable_type = variable_type.rename(columns={'var_name':'v_name','var_dtype':'v_type'})
variable_type = variable_type.set_index(['v_name'])

#加载数据
dataset =ld.load_from_csv(rawdata_path,['af_policy_temp',])
dataset_train = dataset[0]
dataset_train.columns = [col.split('.')[-1] for col in dataset_train.columns]
dataset_len = len(dataset_train)

#改变数据类型
cfd.change_feature_dtype(dataset_train,variable_type)

#定义类及属性，用于结构化输出结果
class InfoValue(object):

    def __init__(self):
        self.var_name = []
        self.split_list = []
        self.iv = 0
        self.woe_list = []
        self.iv_list = []
        self.c2d_trans = []


########分箱处理
bin_var_list = config[config['is_tobe_bin']==1]['var_name']

rst = [] #InfoValue类实例列表
for var in bin_var_list:
    try:
        print 'Running:',var
        a = InfoValue()
        a.var_name = var
        #计算最优分隔点
        a.split = bin.binning_data_split(dataset_train,var)
        # print split

        #合并小组别
        a.split = bin.check_point(dataset_train,var,split,dataset_len)
        print a.split

        #当前实例插入到结果列表中
        rst.extend(a)
        #跟进分隔点区间将连续值替换成分类变量
        # dataset_train[var] = bin.c2d(dataset_train[var],split)
        # print set(dataset_train[var])
    except Exception, e:
        print 'ERROR DATA BIN:',var


#计算IV和WOE替换
candidate_var_list = config[config['is_candidate']==1]['var_name']
iv_list = pd.Series()
for var in candidate_var_list:
    #计算每个变量的IV值
    try:
        iv_list[var] = ci.calculate_iv(dataset_train,var)
        print var,' the iv value: ', iv_list[var]

        # WOE变换
        wt.woe_transformation(dataset_train,var)
    except Exception, e:
        print 'ERROR CALCULATE IV:',var

var_name = 'name_cnt'
split = [0,1,2]


def calculate_split_point_iv(df,var):
    a = df.loc[:,[var,'target']]
    b = a.groupby(['target']).count()
    bt = 1061
    gt = 4484
    bri = b.ix[1,:] * 1.0 / bt
    gri = b.ix[0,:] * 1.0 / gt
    woei = np.log(bri / gri)
    ivi = (bri - gri) * woei
    # print 'woei: ',woei
    # print 'ivi: ',ivi
    return (woei,ivi)


def calculate_split_iv(df,var,split_list):
    c = {'var_name': [], 'var_iv':{'iv': [], 'split_list': [], 'woe_list': [], 'iv_list': []}}
    c['var_name'] = var
    c['var_iv']['split_list'] = split_list
    dfcp = df[:]
    woei = 0
    ivi = 0
    for i in range(0, len(split_list)):
        dfi = dfcp[dfcp[var] <= split_list[i]]
        dfcp = dfcp[dfcp[var] > split_list[i]]
        woei,ivi = calculate_split_point_iv(dfi, var)
        c['var_iv']['woe_list'].extend(woei)
        c['var_iv']['iv_list'].extend(ivi)

    woei,ivi = calculate_split_point_iv(dfcp, var)
    c['var_iv']['woe_list'].extend(woei)
    c['var_iv']['iv_list'].extend(ivi)
    c['var_iv']['iv'] = sum(c['var_iv']['iv_list'])
    return c

def calculate_split_iv2(df,var,split_list):
    c = {'var_name': [], 'iv': [], 'split_list': [], 'woe_list': [], 'iv_list': []}
    c['var_name'] = var
    c['split_list'] = split_list
    dfcp = df[:]
    woei = 0
    ivi = 0
    for i in range(0, len(split_list)):
        dfi = dfcp[dfcp[var] <= split_list[i]]
        dfcp = dfcp[dfcp[var] > split_list[i]]
        woei,ivi = calculate_split_point_iv(dfi, var)
        c['woe_list'].extend(woei)
        c['iv_list'].extend(ivi)

    woei,ivi = calculate_split_point_iv(dfcp, var)
    c['woe_list'].extend(woei)
    c['iv_list'].extend(ivi)
    c['iv'] = sum(c['iv_list'])
    return c

