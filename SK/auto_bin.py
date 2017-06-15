# -*- coding:utf-8 -*-
__author__ = 'maomaochong'
from sklearn.datasets import load_breast_cancer
data = load_breast_cancer()

import pandas as pd
import numpy as np
import scipy.stats.stats as stats

# define a binning function
def auto_bin(Y, X, n=20):
    # fill missings with median
    X2 = X.fillna(np.median(X))                                         #中位数填充缺失值
    r = 0
    while np.abs(r) < 1:
        d1 = pd.DataFrame({"X": X2, "Y": Y, "Bucket": pd.qcut(X2, n)})  #等分位分箱
        d2 = d1.groupby('Bucket', as_index=True)                        #分组
        r, p = stats.spearmanr(d2.mean().X, d2.mean().Y)                #等级相关系数与P值
        n = n - 1
    d3 = pd.DataFrame(d2.min().X, columns=['min_' + X.name])
    d3['max_' + X.name] = d2.max().X
    d3[Y.name] = d2.sum().Y
    d3['total'] = d2.count().Y
    d3[Y.name + '_rate'] = d2.mean().Y
    d4 = (d3.sort_index(by='min_' + X.name)).reset_index(drop=True)
    print "=" * 60
    print d4


auto_bin(data.target, data.data[:,0])
auto_bin(data.target, data.data[:,1])
auto_bin(data.target, data.data[:,2])

