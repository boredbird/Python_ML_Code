# -*- coding:utf-8 -*-
from sklearn import datasets
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from mpl_toolkits.mplot3d import axes3d
boston = datasets.load_boston()
X = boston.data
y = boston.target
num_features = boston.feature_names.__len__()

"""
划分数据集
"""
x_train, x_test, y_train, y_test = train_test_split(X, y,train_size=0.8,random_state=1)

title = 'boston dataset scatter plot'
plt.style.use('ggplot')
for i in range(num_features):
    plt.subplot(num_features/4+1, 4, i+1)
    plt.xlim(min(X[:,i]), max(X[:,i]))
    plt.scatter(X[:,i], y, s=8,color='gray', marker='o', linewidths=.1,alpha=0.5)
    plt.xlabel(boston.feature_names[i],fontsize=8)
plt.subplots_adjust(hspace=0.5, wspace=0.3)
# plt.tight_layout()
plt.show()

# init theta
theta0 = np.array([0]*num_features)

# 计算损失函数
def computeCost(X, y, theta):
    m = y.size
    J = 0
    h = X.dot(theta)
    J = 1.0 / (2 * m) * (np.sum(np.square(h - y)))
    return J

# 梯度下降
def gradientDescent(X, y, theta, alpha=0.01,num_iters=1500):
    m = y.size
    J_history = []
    for iter in np.arange(num_iters):
        h = X.dot(theta)
        theta = theta - alpha * (1.0 / m) * (X.T.dot(-h + y))
        J_history.append(computeCost(X, y, theta))
    return (theta, J_history)

# 画出每一次迭代和损失函数变化
theta , Cost_J = gradientDescent(X, y,theta=theta0)
print('theta: ',theta.ravel())

plt.plot(Cost_J)
plt.ylabel('Cost J')
plt.xlabel('Iterations')


regr = LinearRegression()
regr.fit(X[:,1].reshape(-1,1), y.ravel())
plt.plot(y.ravel(), regr.intercept_+regr.coef_*X, label='Linear regression (Scikit-learn GLM)')


