''''' 
Created on Dec 13, 2012 
 
@Author: Dennis Wu 
@E-mail: hansel.zh@gmail.com 
@Homepage: http://blog.csdn.net/wuzh670 
 
Data set download from : http://www.grouplens.org/system/files/ml-100k.zip 
'''  
  
from operator import itemgetter, attrgetter  
from math import sqrt  
import random  
  
def load_data():  
      
    train = {}  
    test = {}  
    filename_train = 'data/ua.base'  
    filename_test = 'data/ua.test'  
      
    for line in open(filename_train):  
        (userId, itemId, rating, timestamp) = line.strip().split('\t')  
        train.setdefault(userId,{})  
        train[userId][itemId] = float(rating)  
    
    for line in open(filename_test):  
        (userId, itemId, rating, timestamp) = line.strip().split('\t')  
        test.setdefault(userId,{})  
        test[userId][itemId] = float(rating)  
          
    return train, test  
  
def calMean(train):  
    stat = 0  
    num = 0  
    for u in train.keys():  
        for i in train[u].keys():  
            stat += train[u][i]  
            num += 1  
    mean = stat*1.0/num  
    return mean  
 
 #feature 特征值（特征向量，主成分）个数
def initialFeature(feature, userNum, movieNum):  
  
    random.seed(0)  
    user_feature = {}  
    item_feature = {}  
    i = 1  
    while i < (userNum+1):  
        si = str(i)  
        user_feature.setdefault(si,{})  
        j = 1  
        while j < (feature+1):  
            sj = str(j)  
            user_feature[si].setdefault(sj,random.uniform(0,1))  
            j += 1  
        i += 1  
      
    i = 1  
    while i < (movieNum+1):  
        si = str(i)  
        item_feature.setdefault(si,{})  
        j = 1  
        while j < (feature+1):  
            sj = str(j)  
            item_feature[si].setdefault(sj,random.uniform(0,1))  
            j += 1  
        i += 1  
    return user_feature, item_feature  

#user_feature,item_feature都为字典形式  
def svd(train, test, userNum, movieNum, feature, user_feature, item_feature):  
  
    gama = 0.02  
    lamda = 0.3  
    slowRate = 0.99  
    step = 0  
    preRmse = 1000000000.0  
    nowRmse = 0.0  
      
    while step < 100:  
        rmse = 0.0  
        n = 0  
        for u in train.keys():  
            for i in train[u].keys():  
                pui = 0  
                k = 1  
                while k < (feature+1):  
                    sk = str(k)  
                    pui += user_feature[u][sk] * item_feature[i][sk]  #pui为svd估值
                    k += 1  
                eui = train[u][i] - pui  #train[u][i]为 u这个人在i物品上的打分
                rmse += pow(eui,2)  
                n += 1  
                k = 1  
                while k < (feature+1):  
                    sk = str(k)  
                    user_feature[u][sk] += gama*(eui*item_feature[i][sk] - lamda*user_feature[u][sk])  
                    item_feature[i][sk] += gama*(eui*user_feature[u][sk] - lamda**item_feature[i][sk])  
                    k += 1  
              
        nowRmse = sqrt(rmse*1.0/n)  
        print 'step: %d      Rmse: %s' % ((step+1), nowRmse)  
        if (nowRmse < preRmse):  
            preRmse = nowRmse  
              
        gama *= slowRate  
        step += 1  
          
    return user_feature, item_feature  
  
def calRmse(test, user_feature, item_feature, feature):  
      
    rmse = 0.0  
    n = 0  
    for u in test.keys():  
        for i in test[u].keys():  
            pui = 0  
            k = 1  
            while k < (feature+1):  
                sk = str(k)  
                pui += user_feature[u][sk] * item_feature[i][sk]  
                k += 1  
            eui = pui - test[u][i]  
            rmse += pow(eui,2)  
            n += 1  
    rmse = sqrt(rmse*1.0 / n)  
    return rmse;  
     
if __name__ == "__main__":  
  
    # load data  
    train, test = load_data()  
    print 'load data success'  
  
    # initial user and item feature, respectly  
    user_feature, item_feature = initialFeature(100, 943, 1682)  
    print 'initial user and item feature, respectly success'  
      
    # baseline + svd + stochastic gradient descent  
    user_feature, item_feature = svd(train, test, 943, 1682, 100, user_feature, item_feature)  
    print 'svd + stochastic gradient descent success'  
      
    # compute the rmse of test set  
    print 'the Rmse of test test is: %s' % calRmse(test, user_feature, item_feature, 100)  