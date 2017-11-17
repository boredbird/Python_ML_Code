# -*- coding:utf-8 -*-
"""
手撸KNN from 《机器学习实战》
"""
from numpy import *
import operator
from os import listdir

def createDataSet():
    """
    创建数据集
    :return:特征数据 array，标注数据 list
    """
    group = array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
    labels = ['A', 'A', 'B', 'B']
    return group, labels

def autoNorm(dataSet):
    # 参数0使得函数可以从列中选取最小值，而不是选取当前行的最小值
    minVals = dataSet.min(0)
    maxVals = dataSet.max(0)
    ranges = maxVals - minVals
    normDataSet = zeros(shape(dataSet))
    m = dataSet.shape[0]
    normDataSet = dataSet - tile(minVals, (m, 1))
    # element wise divide 矩阵数值相除
    normDataSet = normDataSet / tile(ranges, (m, 1))
    return normDataSet, ranges, minVals

def knn_classifier(inX, dataSet, labels, k):
    """
    k-近邻算法
    :param inX:待分类的输入向量
    :param dataSet:训练样本集
    :param labels:标签向量
    :param k:用于选择最近邻居的数目
    :return:k个近邻投票的最终类别
    """
    dataSetSize = dataSet.shape[0]
    # 计算两个向量点的距离
    # tile: 复制输出，将变量内容复制成输入矩阵同样大小的矩阵
    diffMat = tile(inX, (dataSetSize, 1)) - dataSet
    sqDiffMat = diffMat ** 2
    sqDistances = sqDiffMat.sum(axis=1)
    distances = sqDistances ** 0.5
    sortedDistIndicies = distances.argsort()
    # 选择距离最小的k个点
    classCount = {}
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1
    # 排序
    sortedClassCount = sorted(classCount.iteritems(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]

