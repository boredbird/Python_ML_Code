# -*- coding: utf-8 -*-
"""
Created on Wed Nov 23 16:03:51 2016

@author: zhangchunhui
"""

#coding:UTF-8

from __future__ import division
from math import exp
from numpy import *
from random import normalvariate#正态分布
from datetime import datetime

trainData = 'E://data//diabetes_train.txt'
testData = 'E://data//diabetes_test.txt'
featureNum = 8

def loadDataSet(data):
    dataMat = []
    labelMat = []
    
    fr = open(data)#打开文件
    
    for line in fr.readlines():
        currLine = line.strip().split()
        #lineArr = [1.0]
        lineArr = []
        
        for i in xrange(featureNum):
            lineArr.append(float(currLine[i + 1]))
        dataMat.append(lineArr)
        
        labelMat.append(float(currLine[0]) * 2 - 1)
    return dataMat, labelMat

def sigmoid(inx):
    return 1.0 / (1 + exp(-inx))

def stocGradAscent(dataMatrix, classLabels, k, iter):
    #dataMatrix用的是mat, classLabels是列表
    m, n = shape(dataMatrix)
    alpha = 0.01
    #初始化参数
    w = zeros((n, 1))#其中n是特征的个数
    w_0 = 0.
    v = normalvariate(0, 0.2) * ones((n, k))
    
    for it in xrange(iter):
        print(it)
        for x in xrange(m):#随机优化，对每一个样本而言的
            inter_1 = dataMatrix[x] * v
            inter_2 = multiply(dataMatrix[x], dataMatrix[x]) * multiply(v, v)#multiply对应元素相乘
            #完成交叉项
            interaction = sum(multiply(inter_1, inter_1) - inter_2) / 2.
            
            p = w_0 + dataMatrix[x] * w + interaction#计算预测的输出
        
            loss = sigmoid(classLabels[x] * p[0, 0]) - 1
            print(loss)
        
            w_0 = w_0 - alpha * loss * classLabels[x]
            
            for i in xrange(n):
                if dataMatrix[x, i] != 0:
                    w[i, 0] = w[i, 0] - alpha * loss * classLabels[x] * dataMatrix[x, i]
                    for j in xrange(k):
                        v[i, j] = v[i, j] - alpha * loss * classLabels[x] * (dataMatrix[x, i] * inter_1[0, j] - v[i, j] * dataMatrix[x, i] * dataMatrix[x, i])
        
    
    return w_0, w, v

def getAccuracy(dataMatrix, classLabels, w_0, w, v):
    m, n = shape(dataMatrix)
    allItem = 0
    error = 0
    result = []
    for x in xrange(m):
        allItem += 1
        inter_1 = dataMatrix[x] * v
        inter_2 = multiply(dataMatrix[x], dataMatrix[x]) * multiply(v, v)#multiply对应元素相乘
        #完成交叉项
        interaction = sum(multiply(inter_1, inter_1) - inter_2) / 2.
        p = w_0 + dataMatrix[x] * w + interaction#计算预测的输出
        
        pre = sigmoid(p[0, 0])
        
        result.append(pre)
        
        if pre < 0.5 and classLabels[x] == 1.0:
            error += 1
        elif pre >= 0.5 and classLabels[x] == -1.0:
            error += 1
        else:
            continue
        
    
    print(result)
    
    return float(error) / allItem
        
   
if __name__ == '__main__':
    dataTrain, labelTrain = loadDataSet(trainData)
    dataTest, labelTest = loadDataSet(testData)
    date_startTrain = datetime.now()
    print("开始训练")
    w_0, w, v = stocGradAscent(mat(dataTrain), labelTrain, 20, 200)
    print("训练准确性为：%f" % (1 - getAccuracy(mat(dataTrain), labelTrain, w_0, w, v)))
    date_endTrain = datetime.now()
    print("训练时间为：%s" % (date_endTrain - date_startTrain))
    print("开始测试")
    print("测试准确性为：%f" % (1 - getAccuracy(mat(dataTest), labelTest, w_0, w, v)))  
