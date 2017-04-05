#coding:utf-8
from numpy import *
import operator

def classify0(inX, dataSet, labels, k):
    dataSetSize = dataSet.shape[0]
    diffMat = tile(inX, (dataSetSize,1)) - dataSet
    sqDiffMat = diffMat**2
    sqDistances = sqDiffMat.sum(axis=1)
    distances = sqDistances**0.5
    sortedDistIndicies = distances.argsort()
    classCount={}
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel,0) + 1
    sortedClassCount = sorted(classCount.iteritems(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]

def autoNorm(dataSet):
    minVals = dataSet.min(0)
    maxVals = dataSet.max(0)
    ranges = maxVals - minVals
    normDataSet = zeros(shape(dataSet))
    m = dataSet.shape[0]
    normDataSet = dataSet - tile(minVals, (m,1))
    normDataSet = normDataSet/tile(ranges, (m,1)) #element wise divide
    return normDataSet, ranges, minVals

def file2matrix(filename):
    fr = open(filename)
    numberOfLines = len(fr.readlines()) #get the number of lines in the file
    returnMat = zeros((numberOfLines,37)) #prepare matrix to return
    classLabelVector = [] #prepare labels return
    fr = open(filename)
    index = 0
    for line in fr.readlines():
        line = line.strip()
        listFromLine = line.split(',')
        returnMat[index,:] = listFromLine[0:37]
        classLabelVector.append(int(listFromLine[-1]))
        index += 1
    fr.close()
    return returnMat,classLabelVector

def genderClassTest():
    hoRatio = 0.10 #hold out 10%
    datingDataMat,datingLabels = file2matrix('doubanMovieDataSet.txt') #load data setfrom file
    normMat,ranges,minVals=autoNorm(datingDataMat)
    m = normMat.shape[0]
    numTestVecs = int(m*hoRatio)
    testMat=normMat[0:numTestVecs,:]
    trainMat=normMat[numTestVecs:m,:]
    trainLabels=datingLabels[numTestVecs:m]
    k=3
    errorCount = 0.0
    for i in range(numTestVecs):
        classifierResult = classify0(testMat[i,:],trainMat,trainLabels,k)
        print "the classifier came back with: %d, the real answer is: %d" % (classifierResult, datingLabels[i])
        if (classifierResult != datingLabels[i]):
            errorCount += 1.0
    print "Total errors:%d" %errorCount
    print "The total accuracy rate is %f" %(1.0-errorCount/float(numTestVecs))
    
