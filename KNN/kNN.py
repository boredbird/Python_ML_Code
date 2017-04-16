#机器学习实战练习文档-Chapter02
from numpy import *
import operator

#创建数据集与标签
def createDataSet():
    group=array([[1.0,1.1],[1.0,1.0],[0,0],[0,0.1]])
    labels=['A','A','B','B']
    return group,labels

#对未知类别属性的数据集中的每个点依次执行以下操作：
'''
1、计算已知类别数据集中的点与当前点之间的距离；
2、按照距离递增次序排序；
3、选取与当前点距离最小的k个点；
4、确定前k个点所在类别的出现频率；
5、返回前k个点出现频率最高的类别作为当前点的预测分类。
'''
def classify0(inX,dataSet,labels,k):
    dataSetSize=dataSet.shape[0]
    diffMat=tile(inX,(dataSetSize,1))-dataSet #作差
    sqDiffMat=diffMat**2 #平方
    sqDistances=sqDiffMat.sum(axis=1) #求和
    distances=sqDistances**0.5 #开根号
    sortedDistIndicies=distances.argsort()#升序排序
    classCount=()
    for i in range(k):
        voteIlabel=labels[sortedDistIndicies[i]]
        classCount[voteIlabel]=classCount.get(voteIlabel,0)+1 #将classCount字典分解为元组列表
        sortedClassCount=sorted(classCount.iteritems(),
                                key=operator.itemgetter(1),
                                reverse=True)#逆序排序
        return sortedClassCount[0][0]
    #AttributeError: 'tuple' object has no attribute 'get'

'''
#将文本记录到转换NumPy的解析程序
def file2matrix(filename):
    fr=open(filename)
    arrayOfLines=fr.readlines() #打开文件
    numberOfLines=len(arrayOfLines) #得到文件的行数
    returnMat=zeros((numberOfLines,3)) #创建以0填充的矩阵，为了简化，将矩阵的另一维度设置为固定值3
    classLabelVector=[]
    index=0
    for line in arrayOfLines： 
        line=line.strip() #截取掉所有的回车字符
        listFromLine=line.split('\t') #使用tab字符将上一步得到的整行数据分割成一个元素列表
        returnMat[index,:]=listFromLine[0:3] #选取前3个元素
        classLabelVector.append(int(listFromLine[-1])) #最后一个元素
        index+=1
    return returnMat,classLabelVector
'''


def file2matrix(filename):
    fr = open(filename)
    numberOfLines = len(fr.readlines())         #get the number of lines in the file
    returnMat = zeros((numberOfLines,3))        #prepare matrix to return
    classLabelVector = []                       #prepare labels return   
    fr = open(filename)
    index = 0
    for line in fr.readlines():
        line = line.strip()
        listFromLine = line.split('\t')
        returnMat[index,:] = listFromLine[0:3]
        classLabelVector.append(int(listFromLine[-1]))
        index += 1
    return returnMat,classLabelVector


#归一化特征值
def autoNorm(dataSet):
    minVals=dataSet.min(0)#参数0使得函数可以从列中选取最小值
    maxVals=dataSet.max(0)
    ranges=maxVals-minVals
    normDataSet=zeros(shape(dataSet))
    m=dataSet.shape[0]
    normDataSet=dataSet-tile(minVals,(m,1))#NumPy库中tile()函数将变量内容复制成输入矩阵同样大小的矩阵
    normDataSet=normDataSet/tile(ranges,(m,1))
    return normDataSet,ranges,minVals


#分类器的错误率测试代码
def datingClassTest():
    hoRatio=0.10
    datingDataMat,datingLabels=file2matrix('datingTestSet.txt') #读取文件
    normMat,ranges,minVals=autoNorm(datingDataMat) #归一化
    m=normMat.shape[0]
    numTestVecs=int(m*hoRatio) #测试集数据量
    errorCount=0.0
    for i in range(numTestVecs):
        classfierResult=classify0(normMat[i,:],normMat[numTestVecs:m,:],\
                                  datingLabels[numTestVecs:m],3)
        print "the classifer came back with:%d,the real answer is :%d"\
              %(classifierResult,datingLabels[i])
        if(classiferResult !=datingLabels[i]):errorCount+=1.0 #计算错误量
    print "the total error rate is :%f" % (errorCount/float(numTestVecs)) #错误率


#用分类器进行预测
def classifyPerson():
    resultList=['not at all','in small doses','in large doses']
    percentTats=float(raw_input(\"percentage of time spent playing video games?"))
    ffMiles=float(raw_input("frequent flier miles earned per year?"))
    iceCream=float(raw_input("liters of ice cream consumed per year?"))
    datingDataMat,datingLabels=file2matrix('datingTestSet2.txt')
    normMat,ranges,minVals=autoNorm(datingDataMat)
    inArr=array([ffMiles,percentTats,iceCream])
    classiferResult=classify0((inArr-minVals)/ranges,normMat,datingLabels,3)
    print "You will probably like this person:",resultList[classifierResult-1]
