# -*- coding:utf-8 -*-
"""
手撸决策树 from 《机器学习实战》
"""

from math import log
import operator

# 创建数据集
def createDataSet():
    dataSet = [[1, 1, 'yes'],
               [1, 1, 'yes'],
               [1, 0, 'no'],
               [0, 1, 'no'],
               [0, 1, 'no']]
    labels = ['no surfacing', 'flippers']
    # change to discrete values
    return dataSet, labels

# 计算给定数据集的香农熵
def calcShannonEnt(dataSet):
    numEntries = len(dataSet)
    labelCounts = {} # 为所有可能分类创建字典
    # 循环遍历数据集的每一行
    for featVec in dataSet:  # the the number of unique elements and their occurance
        currentLabel = featVec[-1] # 假设最后一列是label
        if currentLabel not in labelCounts.keys():
            labelCounts[currentLabel] = 0
        labelCounts[currentLabel] += 1
    shannonEnt = 0.0
    for key in labelCounts:
        prob = float(labelCounts[key]) / numEntries
        shannonEnt -= prob * log(prob, 2)  # log base 2
    return shannonEnt

# 按照给定特征划分数据集
def splitDataSet(dataSet, axis, value):
    retDataSet = []
    for featVec in dataSet:
        if featVec[axis] == value: # 只能筛选离散取值的变量
            reducedFeatVec = featVec[:axis]  # chop out axis used for splitting
            reducedFeatVec.extend(featVec[axis + 1:])
            retDataSet.append(reducedFeatVec)
    return retDataSet


# 选择最好的数据集划分方式
def chooseBestFeatureToSplit(dataSet):
    """
    选择最好的数据集划分方式
    :param dataSet:
    :return: 最佳分割变量所在的下标
    """
    numFeatures = len(dataSet[0]) - 1  # the last column is used for the labels
    baseEntropy = calcShannonEnt(dataSet) #计算分割之前的熵
    bestInfoGain = 0.0
    bestFeature = -1
    for i in range(numFeatures):  # iterate over all the features
        # 笨拙的筛选数据集的某一列
        # create a list of all the examples of this feature
        featList = [example[i] for example in dataSet]
        # get a set of unique values
        uniqueVals = set(featList)
        newEntropy = 0.0
        for value in uniqueVals: # 遍历所有取值作为可能的分割点,每个离散值作为一个单独的分组
            subDataSet = splitDataSet(dataSet, i, value) # 只能筛选离散取值的变量
            prob = len(subDataSet) / float(len(dataSet))
            newEntropy += prob * calcShannonEnt(subDataSet)
        infoGain = baseEntropy - newEntropy  # calculate the info gain; ie reduction in entropy
        if (infoGain > bestInfoGain):  # compare this to the best gain so far
            bestInfoGain = infoGain  # if better than current best, set to best
            bestFeature = i
    return bestFeature  # returns an integer

# 计算出现次数最多的分类
def majorityCnt(classList):
    """
    采用多数表决得方式确定该叶子节点的分类
    :param classList: 叶子节点的所有样本标签
    :return: 投票的最终类别
    """
    classCount = {}
    for vote in classList:
        if vote not in classCount.keys(): classCount[vote] = 0
        classCount[vote] += 1
    sortedClassCount = sorted(classCount.iteritems(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]

# 创建树的函数代码
def createTree(dataSet, labels):
    classList = [example[-1] for example in dataSet] # 标签列
    if classList.count(classList[0]) == len(classList):
        return classList[0]  # stop splitting when all of the classes are equal
    if len(dataSet[0]) == 1:  # stop splitting when there are no more features in dataSet
        return majorityCnt(classList)
    bestFeat = chooseBestFeatureToSplit(dataSet) # 最佳分割变量的下标
    bestFeatLabel = labels[bestFeat]
    myTree = {bestFeatLabel: {}}
    del (labels[bestFeat]) # 从候选集中删除当前分割变量
    featValues = [example[bestFeat] for example in dataSet]
    uniqueVals = set(featValues)
    for value in uniqueVals:
        subLabels = labels[:]  # copy all of labels, so trees don't mess up existing labels
        # 在当前节点的所有可能取值上递归继续分割
        # 字典的key:特征名称、特征取值；字典的value: 字典本身 或者 叶节点的类别名称
        myTree[bestFeatLabel][value] = createTree(splitDataSet(dataSet, bestFeat, value), subLabels)
    return myTree


# 使用决策树进行分类
def classify(inputTree, featLabels, testVec):
    """
    递归决策树字典，输出所在叶节点的类别
    :param inputTree: 树结构以字典的形式存储
    :param featLabels: 特征列表
    :param testVec: 待预测的数据集
    :return: 节点类别
    """
    firstStr = inputTree.keys()[0]
    secondDict = inputTree[firstStr]
    # 将标签字符串转换为索引
    featIndex = featLabels.index(firstStr)
    key = testVec[featIndex]
    valueOfFeat = secondDict[key]
    if isinstance(valueOfFeat, dict):
        # 如果当前节点的value仍是一个dict则继续递归
        classLabel = classify(valueOfFeat, featLabels, testVec)
    else:
        # 叶节点
        classLabel = valueOfFeat
    return classLabel

dataSet, labels = createDataSet()
clf_tree = createTree(dataSet, labels)