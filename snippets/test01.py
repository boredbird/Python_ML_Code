# -*- coding:utf-8 -*-
from sklearn.preprocessing import PolynomialFeatures
from sklearn.datasets import load_iris

# 导入IRIS数据集
iris = load_iris()

# 特征矩阵
# iris.data

# 目标向量
# iris.target

#多项式转换
#参数degree为度，默认值为2
# a = PolynomialFeatures().fit_transform(iris.data)

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

#选择K个最好的特征，返回选择特征后的数据
SelectKBest(chi2, k=2).fit_transform(iris.data, iris.target)
