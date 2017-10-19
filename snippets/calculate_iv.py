# -*- coding:utf-8 -*-
__author__ = 'chunhui.zhang'
import numpy as np

def calculate_iv(df,var):
    """
    transformate the discrete var to woe value by the given split_list
    :param df:
    :param var:
    :param split_list:
    :return:
    """
    #todo 分类变量，（和连续变量取值类别小于8的：这个可以用check_point方法处理）的合并策略，分组组内样本量过小和单一因变量取值问题
    a = df.loc[:,[var,'target']]
    b = a.groupby([var, 'target'])['target'].count()
    #todo 某个取值类别只有一个target分类
    # bt = sum(b[:,1])
    # gt = sum(b[:,0])
    bt = 1061
    gt = 4484
    iv = 0
    for i in set(list(df[var])):
        # print i
        bri = b[i,1]*1.0/bt
        # print bri
        gri = b[i,0]*1.0/gt
        # print gri
        woei = np.log(bri/gri)
        # print woei
        iv += (bri-gri)*woei

    return iv

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

# df = dataset_train
# var = 'active_loan_cnt_pos'
# ci.calculate_iv(dataset_train,'active_loan_cnt_pos')
