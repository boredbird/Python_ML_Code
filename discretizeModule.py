#coding=utf-8
import operator
from math import log
import time

class InformationGainSplitDiscretization(object):

	def __init__(self):
		self.minInfoGain_epos = 1e-8   #停止条件之一：最小信息增益，当某数据集的最优分裂对应的信息增益（即最大信息增益）小于这个值，则此数据集停止进一步的分裂。      
		self.splitPiontsList = []     #分裂点列表，最终要依分裂点的值升序排列。以便后续的离散化函数（输入：待离散的数据集）使用。		#self.totalGain = ()
		self.tree_deep = 3

		
	def splitDataSet(self,dataSet, splitpoint_idx):
		leftSubDataSet = []
		rightSubDataSet = []
		for leftSubSet in dataSet[:(splitpoint_idx+1)]:
			leftSubDataSet.append(leftSubSet)

		for rightSubSet in dataSet[(splitpoint_idx+1):]:
			rightSubDataSet.append(rightSubSet)

		leftSubDataSet.sort(key=lambda x : x[0], reverse=False)
		rightSubDataSet.sort(key=lambda x : x[0], reverse=False)
		
		return (leftSubDataSet,rightSubDataSet)


	def calcInfoGain(self,dataSet):
		lable1_sum = 0
		total_sum = 0
		infoGain = 0
		if dataSet == []:
			pass
		else :
			for i in range(len(dataSet)):
				lable1_sum += dataSet[i][1]
				total_sum += dataSet[i][1] + dataSet[i][2]
				
			p1 = lable1_sum / total_sum
			p0 = 1 - p1
			if p1 == 0 or p0 == 0:
				infoGain = 0
			else:
				infoGain = - p0 * log(p0) - p1 * log(p1)

		return infoGain,total_sum
		

	def getMaxInfoGain(self,dataSet):
		gainList = []
		totalGain = self.calcInfoGain(dataSet)
		maxGain = 0
		maxGainIdx = 0 
		for i in range(len(dataSet)):
			leftSubDataSet_info = self.calcInfoGain(self.splitDataSet(dataSet, i)[0])
			rightSubDataSet_info = self.calcInfoGain(self.splitDataSet(dataSet, i)[1])
			gainList.append(totalGain[0] 
			- (leftSubDataSet_info[1]/totalGain[1]) * leftSubDataSet_info[0]
			- (rightSubDataSet_info[1]/totalGain[1]) * rightSubDataSet_info[0])

		maxGain = max(gainList)
		maxGainIdx = gainList.index(max(gainList))
		splitPoint = dataSet[maxGainIdx][0]
		return splitPoint,maxGain,maxGainIdx


	def getSplitPointList(self,dataSet,maxdeeps,begindeep):
		if begindeep >= maxdeeps:
			pass
		else:
			maxInfoGainList = self.getMaxInfoGain(dataSet)
			if maxInfoGainList[1] <= self.minInfoGain_epos: 
				pass
			else:
				self.splitPiontsList.append(maxInfoGainList[0])
				begindeep += 1
				subDataSet = self.splitDataSet(dataSet, maxInfoGainList[2])
				self.getSplitPointList(subDataSet[0],maxdeeps,begindeep)
				self.getSplitPointList(subDataSet[1],maxdeeps,begindeep)
				

	def fit(self, x, y,deep = 3, epos = 1e-8):
		self.minInfoGain_epos = epos
		self.tree_deep = deep		
		bin_dict = {}  
		bin_list = []  
		for i in range(len(x)):  
			pos = x[i] 
			target = y[i]  
			bin_dict.setdefault(pos,[0,0])             
			if target == 1:  
				bin_dict[pos][0] += 1                  
			else:  
				bin_dict[pos][1] += 1  

		for key ,val in bin_dict.items():  
			t = [key]  
			t.extend(val)  
			bin_list.append(t)

		bin_list.sort( key=lambda x : x[0], reverse=False)
		self.getSplitPointList(bin_list,self.tree_deep,0)
		self.splitPiontsList = [elem for elem in self.splitPiontsList if elem != []]
		self.splitPiontsList.sort()


	def transform(self,x):
		res = []
		for e in x :
			index = self.get_Discretization_index(self.splitPiontsList, e)
			res.append(index)

		return res


	def get_Discretization_index(self, Discretization_vals, val):
		index = len(Discretization_vals) + 1
		for i in range(len(Discretization_vals)):
			bin_val = Discretization_vals[i]
			if val <= bin_val:
				index = i + 1
				break

		return index



