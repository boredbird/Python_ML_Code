#ch03
import operator

#计算给定数据集的熵
from math import log
def calcShannonEnt(dataSet):
    numEntries=len(dataSet) #计算数据集中实例的总数
    labelCounts={}
    for featVec in dataSet:
        currentLabel=featVec[-1] #创建一个数据字典，每个键值都记录了当前类别出现的次数
        if currentLabel not in labelCounts.keys():
            labelCounts[currentLabel]=0 #如果当前键值不存在，则扩展字典并将当前键值加入字典
            labelCounts[currentLabel]+=1
        shannonEnt=0.0
        for key in labelCounts:
            prob=float(labelCounts[key])/numEntries #使用所有类标签的发生频率计算类别出现的概率
            shannonEnt -= prob*log(prob,2)#计算香农熵
        return shannonEnt
    

def createDataSet():
    dataSet=[[1,1,'yes'],[1,1,'yes'],[1,0,'no'],[0,1,'no'],[0,1,'no']]
    labels=['no surfacing','flippers']
    return dataSet,labels

#按照给定特征划分数据集
def splitDataSet(dataSet,axis,value):#三个输入参数：待划分的数据集、划分数据集的特征、特征的返回值
    retDataSet=[] #创建新的list对象
    for featVec in dataSet:
        if featVec[axis]==value: #第axis+1个特征值等于value，则追加提取
            reducedFeatVec=featVec[:axis]
            reducedFeatVec.extend(featVec[axis+1:])
            retDataSet.append(reducedFeatVec) #将符合特征的数据抽取出来
    return retDataSet


#遍历整个数据集，循环计算熵值和splitDataSet()函数，找到最好的特征划分方式
#选择最好的数据集划分方式
def chooseBestFeatureToSplit(dataSet):
    numFeatures=len(dataSet[0])-1
    baseEntropy=calcShannonEnt(dataSet) #计算整个数据集的原始香农熵
    bestInfoGain=0.0;bestFeature=-1
    for i in range(numFeatures): #遍历数据集中所有特征
        featList=[example[i] for example in dataSet] #将数据集中所有第i个特征值或者所有可能存在的值写入到这个list中
        uniqueVals=set(featList) #集合(set)数据类型
        newEntropy=0.0
        for value in uniqueVals: #遍历当前特征中的所有唯一属性值，对每个特征划分一次数据集
            subDataSet=splitDataSet(dataSet,i,value)
            prob=len(subDataSet)/float(len(dataSet))
            newEntropy +=prob*calcShannonEnt(subDataSet) #计算数据集的新熵值
        infoGain =baseEntropy-newEntropy #计算信息增益
        if (infoGain > bestInfoGain): #比较所有特征中的信息增益
            bestInfoGain =infoGain
            bestFeature =i
    return bestFeature #返回最好特征划分的索引值

'''
递归构建决策树
递归结束的条件是：程序遍历完所有划分数据集的属性，或者每个分支下的所有实例都具有相同的分类。
如果所有实例具有相同的分类，则得到一个叶子节点或者终止块。
'''
#采用多数表决的方法决定该叶子节点的分类
def majorityCnt(classList):
    classCount =[]
    for vote in classList:
        if vote not in classCount.keys():classCount[vote] = 0
        classCount[vote] +=1
        sortedClassCount =sorted(classCount.iteritems(),
                                 key=operator.itemgetter(1),
                                 reverse=True)
        return sortedClassCount[0][0]

#创建树
def createTree(dataSet,labels):
    classList =[example[-1] for example in dataSet]
    if classList.count(classList[0]) == len(classList):
        return classList[0]
    if len(dataSet[0]) ==1:
        return majorityCnt(classList)
    bestFeat =chooseBestFeatureToSplit(dataSet)
    bestFeatLabel = labels[bestFeat]
    myTree = [bestFeatLabel:[]]
    del(labels[bestFeat])
    featValues =[example[bestFeat] for example in dataSet]
    uniqueVals = set(featValues)
    for value in uniqueVals:
        subLabels =labels[:]
        myTree[bestFeatLabel][value] =createTree(splitDataSet\
                                                 (dataSet,bestFeat,value),subLabels)
    return myTree
