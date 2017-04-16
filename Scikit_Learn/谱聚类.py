#coding:UTF-8  
''''' 
Created on 2015��5��12�� 
 
@author: zhaozhiyong 
'''  
from __future__ import division  
import scipy.io as scio  
from scipy import sparse  
from scipy.sparse.linalg.eigen import arpack#����ֻ����ô������Ȼʼ���Ҳ�������eigs  
from numpy import *  
  
  
def spectalCluster(data, sigma, num_clusters):  
    print "���ڽӾ���ת�������ƾ���"  
    #�����sigma ��= 0  
    print "Fixed-sigma�׾���"  
    data = sparse.csc_matrix.multiply(data, data)  
  
    data = -data / (2 * sigma * sigma)  
      
    S = sparse.csc_matrix.expm1(data) + sparse.csc_matrix.multiply(sparse.csc_matrix.sign(data), sparse.csc_matrix.sign(data))     
      
    #ת����Laplacian����  
    print "�����ƾ���ת����Laplacian����"  
    D = S.sum(1)#���ƾ����ǶԳƾ���  
    D = sqrt(1 / D)  
    n = len(D)  
    D = D.T  
    D = sparse.spdiags(D, 0, n, n)  
    L = D * S * D  
      
    #������ֵ����������  
    print "������ֵ����������"  
    vals, vecs = arpack.eigs(L, k=num_clusters,tol=0,which="LM")    
      
    # ����k-Means  
    print "����K-Means��������������"  
    #��vecs�����滯  
    sq_sum = sqrt(multiply(vecs,vecs).sum(1))  
    m_1, m_2 = shape(vecs)  
    for i in xrange(m_1):  
        for j in xrange(m_2):  
            vecs[i,j] = vecs[i,j]/sq_sum[i]  
      
    myCentroids, clustAssing = kMeans(vecs, num_clusters)  
      
    for i in xrange(shape(clustAssing)[0]):  
        print clustAssing[i,0]  
      
  
def randCent(dataSet, k):  
    n = shape(dataSet)[1]  
    centroids = mat(zeros((k,n)))#create centroid mat  
    for j in range(n):#create random cluster centers, within bounds of each dimension  
        minJ = min(dataSet[:,j])   
        rangeJ = float(max(dataSet[:,j]) - minJ)  
        centroids[:,j] = mat(minJ + rangeJ * random.rand(k,1))  
    return centroids  
  
def distEclud(vecA, vecB):  
    return sqrt(sum(power(vecA - vecB, 2))) #la.norm(vecA-vecB)  
  
def kMeans(dataSet, k):  
    m = shape(dataSet)[0]  
    clusterAssment = mat(zeros((m,2)))#create mat to assign data points to a centroid, also holds SE of each point  
    centroids = randCent(dataSet, k)  
    clusterChanged = True  
    while clusterChanged:  
        clusterChanged = False  
        for i in range(m):#for each data point assign it to the closest centroid  
            minDist = inf; minIndex = -1  
            for j in range(k):  
                distJI = distEclud(centroids[j,:],dataSet[i,:])  
                if distJI < minDist:  
                    minDist = distJI; minIndex = j  
            if clusterAssment[i,0] != minIndex: clusterChanged = True  
            clusterAssment[i,:] = minIndex,minDist**2  
        #print centroids  
        for cent in range(k):#recalculate centroids  
            ptsInClust = dataSet[nonzero(clusterAssment[:,0].A==cent)[0]]#get all the point in this cluster  
            centroids[cent,:] = mean(ptsInClust, axis=0) #assign centroid to mean   
    return centroids, clusterAssment  
  
  
if __name__ == '__main__':  
    # �������ݼ�  
    matf = 'E://data_sc//corel_50_NN_sym_distance.mat'  
    dataDic = scio.loadmat(matf)  
    data = dataDic['A']  
    # �׾���Ĺ���  
    spectalCluster(data, 20, 18)  