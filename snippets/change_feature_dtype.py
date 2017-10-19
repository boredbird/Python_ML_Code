# -*- coding:utf-8 -*-
__author__ = 'chunhui.zhang'
import sys
sys.path.append(r'E:\Code\ScoreCard')
from config_params import *
import re

def remove_thousands_separator(str_obj):
    """
    将带，逗号分隔的数字中的逗号去掉；
    返回一个object类型的字符串
    :param str_obj:
    :return:
    """
    return re.sub("\D", "", str_obj)


def change_feature_dtype(df,variable_type):
    """
    change feature data type by the variable_type specified in the config_params.py file
    :param df:
    :return:
    """
    if len(df.columns) == variable_type.shape[0]:
        for vname in df.columns:
            try:
                df[vname] = df[vname].astype(variable_type.loc[vname,'v_type'])
                print vname,variable_type.loc[vname,'v_type']
            except:
                print '[error]',vname
        print "Variable dtypes have been specified!"
    else:
        print len(df.columns)
        print variable_type.shape[0]
        raise ValueError("the colums num of dataset_train and varibale_type is not equal")


    return 1