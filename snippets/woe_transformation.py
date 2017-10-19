# -*- coding:utf-8 -*-
__author__ = 'chunhui.zhang'
import numpy as np
import copy

def woe_transformation(df,var):
    """
    transformate the discrete var to woe value by the given split_list
    :param df:
    :param var:
    :param split_list:
    :return:
    """
    # df = dataset_train
    # var = 'pos_finished_periods_cnt'
    a = df.loc[:,[var,'target']]
    b = a.groupby([var, 'target'])['target'].count()

    bt = sum(b[:,1])
    gt = sum(b[:,0])

    df_woe = copy.deepcopy(df)

    for i in set(list(df[var])):
        bri = b[i,1]*1.0/bt
        gri = b[i,0]*1.0/gt
        woei = np.log(bri/gri)
        # print woei
        df_woe.loc[df[var] == i,var] = woei

    return df_woe[var]
