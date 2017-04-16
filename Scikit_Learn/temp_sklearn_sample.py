# -*- coding: utf-8 -*-
"""
Created on Sun Aug 16 17:34:28 2015

@author: Administrator
"""
from sklearn import linear_model
"""
#LinearRegression
from sklearn import linear_model
clf = linear_model.LinearRegression()
clf.fit([[0,0],[1,1],[2,2]],[0,1,2])
clf.coef_
"""

"""
#Ridge Regression
from sklearn import linear_model
clf = linear_model.Ridge(alpha=.5)
clf.fit([[0,0],[0,0],[1,1]],[0,.1,1])
print('clf.coef_ = ',clf.coef_)
print('clf.intercept_ = ',clf.intercept_)
"""


"""
#RidgeCV Regression
from sklearn import linear_model
clf = linear_model.RidgeCV(alpha=[0.1,1.0,10.0])
clf.fit([[0,0],[0,0],[1,1]],[0,.1,1])
print('clf.alpha_ = ',clf.alpha_)
"""


#Least Angle Regression
clf = linear_model.Lasso(alpha=0.1)
clf.fit([[0,0],[0,1],[1,1]])
clf.predict([[1,1]])


