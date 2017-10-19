# -*- coding:utf-8 -*-
__author__ = 'chunhui.zhang'
import sys
sys.path.append(r'E:\Code\ScoreCard')
from config_params import *
import collections
import numpy as np
import copy

def binning_data_by_iv_point(df,var,split_point):
    """
    calculate the iv value with the specified split point
    note:
        the dataset should have variables:'target' which to be encapsulated if have time
    :return:
    """
    #split dataset
    dataset_r = df[df.loc[:,var] >= split_point][[var,'target']]
    dataset_l = df[df.loc[:,var] < split_point][[var,'target']]

    #calculate subset statistical frequency
    a = dataset_r.groupby(['target', ]).count().reset_index()
    a.rename(columns={var:'cnt'}, inplace = True)
    # r0_cnt = a[a['target']==0]['cnt'][:1].real[0]
    r0_cnt = sum(a[a['target']==0]['cnt'])
    # r1_cnt = a[a['target']==1]['cnt'][:1].real[0]
    r1_cnt = sum(a[a['target']==1]['cnt'])

    b = dataset_l.groupby(['target', ]).count().reset_index()
    b.rename(columns={var:'cnt'}, inplace = True)
    # l0_cnt = b[b['target']==0]['cnt'][:1].real[0]
    l0_cnt = sum(b[b['target']==0]['cnt'])
    # l1_cnt = b[b['target']==1]['cnt'][:1].real[0]
    l1_cnt = sum(b[b['target']==1]['cnt'])

    if r0_cnt == 0 or r1_cnt == 0 or l0_cnt == 0 or l1_cnt ==0:
        return 0,0,0,dataset_l,dataset_r
    #calculate woe,iv
    #br aka Bag Ratio,Bi/Bt;gr aka Good Ratio,Gi/Gt;
    #l* or *l named left dataset via the split;r* or *l named right dataset via the split;
    # bt = l1_cnt + r1_cnt
    # gt = l0_cnt + r0_cnt
    bt = 85427
    gt = 885502
    lbr = l1_cnt*1.0/bt
    lgr = l0_cnt*1.0/gt
    woel = np.log(lbr/lgr)
    ivl = (lbr-lgr)*woel
    rbr = r1_cnt*1.0/bt
    rgr = r0_cnt*1.0/gt
    woer = np.log(rbr/rgr)
    ivr = (rbr-rgr)*woer
    iv = ivl+ivr

    return woel,woer,iv,dataset_l,dataset_r


def binning_data_by_iv(df,var,dataset_len,bestSplit=None):
    """
    discretize var into few bins by minimizing the iv
    :param df:
    :param var:
    :return:
    """
    #split the dataset 100 pieces,remove duplicate split point using Func set()
    percent_value = list(set(np.percentile(df[var],range(100))))
    percent_value.sort()

    #if the count of pencent_value is less than 2
    #or the num of subdataset is less than totalset*0.05
    #then dont split the input dataset
    if len(df) < dataset_len*0.05 \
            or percent_value[1:len(percent_value) - 1] == []:
        return bestSplit.woel, bestSplit.woer, bestSplit.iv, bestSplit.point, bestSplit.dataset_l, bestSplit.dataset_r


    #init bestSplit_iv with zero
    bestSplit_iv = 0
    bestSplit_woel = 0
    bestSplit_woer = 0
    bestSplit_point = 0
    bestSplit_dataset_l = pd.DataFrame()
    bestSplit_dataset_r = pd.DataFrame()
    #remove max value and min value in case dataset_r  or dataset_l will be null
    for point in percent_value[1:len(percent_value)-1]:
        # print point
        woel,woer,iv,dataset_r,dataset_l = binning_data_by_iv_point(df,var,point)
        if iv > bestSplit_iv:
            bestSplit_woel = woel
            bestSplit_woer = woer
            bestSplit_iv = iv
            bestSplit_point = point
            bestSplit_dataset_r = dataset_r
            bestSplit_dataset_l = dataset_l

    return bestSplit_woel,bestSplit_woer,bestSplit_iv,bestSplit_point,bestSplit_dataset_l,bestSplit_dataset_r


def binning_data_split(df,var):
    """
    Specify the data split level and return the split value list
    :return:
    """
    if len(set(df[var])) <=8:
        split = list(set(df[var]))
        split.sort()
        return split

    if len(set(np.percentile(df[var],range(100))))==1:
        return list(set(np.percentile(df[var],range(100))))

    bestSplit=collections.namedtuple('bestSplit',['woel','woer','iv','point','dataset_l','dataset_r'])

    # print 'Split Level One:'
    woel,woer,iv,point,dataset_l,dataset_r = binning_data_by_iv(df,var,dataset_len=len(df))
    bestSplit_one = bestSplit(woel,woer,iv,point,dataset_l,dataset_r)

    # print 'Split Level Two 1:'
    woel,woer,iv,point,dataset_l,dataset_r = binning_data_by_iv(bestSplit_one.dataset_l,var,len(df),bestSplit_one)
    bestSplit_two1 = bestSplit(woel,woer,iv,point,dataset_l,dataset_r)
    # print 'Split Level Two 2:'
    woel,woer,iv,point,dataset_l,dataset_r = binning_data_by_iv(bestSplit_one.dataset_r,var,len(df),bestSplit_one)
    bestSplit_two2 = bestSplit(woel,woer,iv,point,dataset_l,dataset_r)

    # print 'Split Level Three 1:'
    woel,woer,iv,point,dataset_l,dataset_r = binning_data_by_iv(bestSplit_two1.dataset_l,var,len(df),bestSplit_two1)
    bestSplit_three1 = bestSplit(woel,woer,iv,point,dataset_l,dataset_r)
    # print 'Split Level Three 2:'
    woel,woer,iv,point,dataset_l,dataset_r = binning_data_by_iv(bestSplit_two1.dataset_r,var,len(df),bestSplit_two1)
    bestSplit_three2 = bestSplit(woel,woer,iv,point,dataset_l,dataset_r)
    # print 'Split Level Three 3:'
    woel,woer,iv,point,dataset_l,dataset_r = binning_data_by_iv(bestSplit_two2.dataset_l,var,len(df),bestSplit_two2)
    bestSplit_three3 = bestSplit(woel,woer,iv,point,dataset_l,dataset_r)
    # print 'Split Level Three 4:'
    woel,woer,iv,point,dataset_l,dataset_r = binning_data_by_iv(bestSplit_two2.dataset_r,var,len(df),bestSplit_two2)
    bestSplit_three4 = bestSplit(woel,woer,iv,point,dataset_l,dataset_r)

    split = [bestSplit_one.point,bestSplit_two1.point,bestSplit_two2.point
        ,bestSplit_three1.point,bestSplit_three2.point,bestSplit_three3.point,bestSplit_three4.point]

    while [] in split:
        split.remove([])

    split = list(set(split))
    split.sort()

    # return the sorted split point
    return  split

def c2d(continuous_var,split_list):
    """
    replace continuous_var to discrete_list
    :param continuous_var:
    :param split_list:
    :return:
    """
    discrete_code = ['A','B','C','D','E','F','G','H','I','J']

    discrete_var = copy.deepcopy(continuous_var)
    #为了避免 dd6_pos:-1 ;小于-1没有值导致只有一个分组的情况
    #将 < 改为 <= ;将 >= 改为 >
    discrete_var[continuous_var <= split_list[0]] = discrete_code[0]
    for i in range(0,len(split_list)-1):
        discrete_var[(continuous_var > split_list[i])*(continuous_var <= split_list[i+1])] = discrete_code[i+1]

    discrete_var[continuous_var > split_list[len(split_list)-1]] = discrete_code[len(split_list)]

    return discrete_var


def check_point(df,var,split,dataset_len):
    """
    检查分割点是否会造成有些分组样本量过小;
    如果存在分组样本量低于总样本量的5%，则与相邻分组合并直至超过5%为止;
    仅适用于连续值
    :param df:
    :param var:
    :param split:
    :return:
    """
    new_split = []
    new_split.append(split[0])
    for i in range(0,len(split)-1):
        if len(df[(df[var] > split[i]) & (df[var] <= split[i+1])]) < dataset_len*0.05:
            continue
        else:
            new_split.append(split[i+1])


    if (len(df[df[var] > split[len(split)-1]])< dataset_len*0.05) & (len(new_split)>1):
        new_split.pop()

    #split只有一个取值，且没有比这个值更小的值，例如dd6_pos:-1
    if new_split == []:
        new_split = split

    return new_split


def factor_binning(df,var):
    """
    #TODO调用calculate_iv,计算每个取值的iv，根据iv值的大小关系归到iv值最接近的分组；
    #相当于df在var方向上经过woe变换后的单个变量聚类分组
    #借鉴决策树后剪枝过程，但是没必要像CART那样形成两个超类
    :param df: 数据框
    :param var: 分类变量
    :return: 返回字典：分类变量的各个取值对应分箱组合后的分类值
    """

    return dict_bin