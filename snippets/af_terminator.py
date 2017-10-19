# -*- coding:utf-8 -*-
__author__ = 'chunhui.zhang'
import sys
sys.path.append(r'E:\Code\ScoreCard')
import numpy as np
from config_params import  *
import load_data as ld

class node:
    '''树的节点的类
    '''
    def __init__(self,var_name=None,iv=0,split_point=None,right=None,left=None):
        self.var_name = var_name  # 用于切分数据集的属性的列索引值
        self.iv = iv  # 设置叶节点的iv值
        self.split_point = split_point  # 存储划分的值
        self.right = right  # 右子树
        self.left = left  # 左子树


#定义类及属性，用于结构化输出结果
class InfoValue(object):

    def __init__(self):
        self.var_name = []
        self.split_list = []
        self.iv = 0
        self.woe_list = []
        self.iv_list = []
        self.c2d_trans = []

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

def check_point(df,var,split,min_sample):
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
        pdf = df[(df[var] > split[i]) & (df[var] <= split[i+1])]
        if (len(pdf) < min_sample) & (len(set(pdf['target']))>1):
            continue
        else:
            new_split.append(split[i+1])

    #剩余样本太少，则去掉最后一个分割掉
    if (len(df[df[var] > split[len(split)-1]])< min_sample) & (len(new_split)>1):
        new_split.pop()
    #剩余样本只有正样本或负样本，则去掉最后一个分割掉
    if len(set(df[df[var] > split[len(split)-1]]['target']))>1:
        new_split.pop()

    #split只有一个取值，且没有比这个值更小的值，例如dd6_pos:-1
    if new_split == []:
        new_split = split

    return new_split

def binning_data_by_iv(df,var,min_sample,bestSplit=None):
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
    if len(df) < min_sample \
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

def calculate_split_point_iv(df,var):
    a = df.loc[:,[var,'target']]
    b = a.groupby(['target']).count()
    bt = 1061
    gt = 4484
    bri = b.ix[1,:] * 1.0 / bt
    gri = b.ix[0,:] * 1.0 / gt
    woei = np.log(bri / gri)
    ivi = (bri - gri) * woei
    return (woei,ivi)

def binning_data_by_iv_point(df,var,split_point):
    """
    calculate the iv value with the specified split point
    note:
        the dataset should have variables:'target' which to be encapsulated if have time
    :return:
    """
    #split dataset
    dataset_r = df[df.loc[:,var] > split_point][[var,'target']]
    dataset_l = df[df.loc[:,var] <= split_point][[var,'target']]

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
        return 0,0,0,dataset_l,dataset_r,0,0
    #calculate woe,iv
    #br aka Bag Ratio,Bi/Bt;gr aka Good Ratio,Gi/Gt;
    #l* or *l named left dataset via the split;r* or *l named right dataset via the split;
    # bt = l1_cnt + r1_cnt
    # gt = l0_cnt + r0_cnt
    bt = 1061
    gt = 4484
    lbr = l1_cnt*1.0/bt
    lgr = l0_cnt*1.0/gt
    woel = np.log(lbr/lgr)
    ivl = (lbr-lgr)*woel
    rbr = r1_cnt*1.0/bt
    rgr = r0_cnt*1.0/gt
    woer = np.log(rbr/rgr)
    ivr = (rbr-rgr)*woer
    iv = ivl+ivr

    return woel,woer,iv,dataset_l,dataset_r,ivl,ivr

def check_target(df):
    """
    检查是否同时存在正负样本
    :param df:
    :param var:
    :return:
    """
    if_pass = 0
    if len(set(df['target']))>1:
        if_pass = 1

    return  if_pass


def deco(func):
    def _deco(*args,**kwargs):
        print("Before binning_data_split() called.")
        func(*args,**kwargs)
        print("After binning_data_split() called.")
        # 不需要返回func，实际上应返回原函数的返回值
    return _deco


@deco
def binning_data_split(df,var,min_sample,iv=0):
    """
    Specify the data split level and return the split value list
    :return:
    """
    sign = 1 #是否继续分割的标识符

    iv_var = InfoValue()

    if len(set(df[var])) <=8:
        split = list(set(df[var]))
        split.sort()
        #分割点检查与处理
        split = check_point(df, var, split, min_sample)
        split.sort()
        iv_var.split_list = split
        #todo 根据split计算iv，独立的函数
        sign = 0 #停止分割

        return node(split_point=split)

    percent_value = list(set(np.percentile(df[var], range(100))))
    percent_value.sort()

    if len(percent_value) <=2:
        iv_var.split_list = list(set(percent_value)).sort()
        sign = 0  # 停止分割
        return node(split_point=percent_value)

    # for point in percent_value[1:len(percent_value)-1]:
    #     # print point
    #     dfcp = df[:]
    #     dfi = dfcp[dfcp[var] <= point]
    #     dfcp = dfcp[dfcp[var] > point]
    #     woei, ivi = calculate_split_point_iv(dfi, var)


    #init bestSplit_iv with zero
    bestSplit_iv = 0
    bestSplit_woel = []
    bestSplit_woer = []
    bestSplit_ivl = 0
    bestSplit_ivr = 0
    bestSplit_point = []
    bestSplit_dataset_l = pd.DataFrame()
    bestSplit_dataset_r = pd.DataFrame()
    #remove max value and min value in case dataset_r  or dataset_l will be null
    for point in percent_value[0:len(percent_value)-1]:
        # print point
        # 只有正样本或负样本，则跳过
        if len(set(df[df[var] > point]['target'])) == 1 or len(set(df[df[var] <= point]['target'])) == 1:
            continue

        woel, woer, iv, dataset_l, dataset_r, ivl, ivr = binning_data_by_iv_point(df,var,point)
        if iv > bestSplit_iv:
            bestSplit_woel = woel
            bestSplit_woer = woer
            bestSplit_iv = iv
            bestSplit_point = point
            bestSplit_dataset_r = dataset_r
            bestSplit_dataset_l = dataset_l
            bestSplit_ivl = ivl
            bestSplit_ivr = ivr

    # iv_var.split_list.extend(bestSplit_point)
    # iv_var.iv_list.extend(bestSplit_iv)
    # iv_var.woe_list.extend([bestSplit_woel,bestSplit_woer])
    # node(fea=bestCriteria[0], value=bestCriteria[1], \
    #      right=right, left=left)

    # self.var_name = []  # 用于切分数据集的属性的列索引值
    # self.iv = 0  # 设置叶节点的iv值
    # self.split_point = []  # 存储划分的值
    # self.right = None  # 右子树
    # self.left = None  # 左子树
    #
    print '当前层级划分完毕！'

    presplit_right = node()
    presplit_left = node()

    print '进入节点 右！'
    if len(bestSplit_dataset_r) < min_sample or len(set(bestSplit_dataset_r['target'])) == 1:
        presplit_right.iv = bestSplit_ivr
        sign = 0  # 停止分割
        print 'presplit_right.iv: ',presplit_right.iv
        right = presplit_right
        # return presplit_right
    else:
        right = binning_data_split(bestSplit_dataset_r, var, min_sample)

    print '进入节点 左！'
    if len(bestSplit_dataset_l) < min_sample or len(set(bestSplit_dataset_l['target'])) == 1:
        presplit_left.iv = bestSplit_ivl
        sign = 0  # 停止分割
        print 'presplit_left.iv: ',presplit_left.iv
        left = presplit_left
        # return presplit_left
    else:
        left = binning_data_split(bestSplit_dataset_l, var, min_sample)

    #判断是否结束
    print 'presplit_right:',presplit_right
    print 'presplit_left:',presplit_left
    if presplit_right.iv + presplit_left.iv <= bestSplit_iv:
        sign = 0  # 停止分割
        return node(var_name=var,iv=bestSplit_iv,split_point=bestSplit_point,right=right,left=left)

    #todo 判断是否结束，反过来写


if __name__ == "__main__":
    rawdata_path = 'E:/ScoreCard/rawdata/'
    raw_file_name = ['af_policy_temp', ]
    train_file_name = 'af_policy_temp'
    # model path
    model_path = 'E:/Code/ScoreCard/'

    config = ld.load_from_csv(model_path, ['config', ])
    config = config[0]

    variable_type = config[['var_name', 'var_dtype']]
    variable_type = variable_type.rename(columns={'var_name': 'v_name', 'var_dtype': 'v_type'})
    variable_type = variable_type.set_index(['v_name'])

    # 加载数据
    dataset = ld.load_from_csv(rawdata_path, ['af_policy_temp', ])
    dataset_train = dataset[0]
    dataset_train.columns = [col.split('.')[-1] for col in dataset_train.columns]
    dataset_len = len(dataset_train)

    # 改变数据类型
    change_feature_dtype(dataset_train, variable_type)

    # 分箱处理
    bin_var_list = config[config['is_tobe_bin'] == 1]['var_name']

    rst = []  # InfoValue类实例列表

    min_sample = 250
    var = 'name_dup_cnt'
    iv_tree = binning_data_split(dataset_train, var, min_sample, iv=0)