# -*- coding:utf-8 -*-
import scipy.stats as stats
import numpy as np
from sklearn import linear_model
x = [[76,81,78,76,76,78,76,78,98,88,76,66,44,67,65,59,87,77,79,85,68,76,77,98,99,98,87,67,78]]
y = [43,33,23,34,31,51,56,43,44,45,32,33,28,39,31,38,21,27,43,46,41,41,48,56,55,45,68,54,33]
x = np.array(x)
r, p=stats.pearsonr(x[0],y)
regr = linear_model.LinearRegression()
regr.fit(x.T, y)

x = [76,81,78,76,76,78,76,78,98,88,76,66,44,67,65,59,87,77,79,85,68,76,77,98,99,98,87,67,78]
y = [43,33,23,34,31,51,56,43,44,45,32,33,28,39,31,38,21,27,43,46,41,41,48,56,55,45,68,54,33]
x.extend(x)
y.extend(y)
stats.pearsonr(x,y)



