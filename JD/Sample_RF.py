# coding=utf-8
#导入数值计算库

import numpy as np

#导入科学计算库

import pandas as pd

#导入交叉验证库

from sklearn import cross_validation

#导入随机森林算法库

from sklearn.ensemble import RandomForestClassifier
#读取流量数据并创建名为traffic的数据表

traffic=pd.DataFrame(pd.read_excel(‘traffic_type.xlsx’))
#查看数据表内容

traffic.head()
#查看数据表列标题

traffic.columns

Index([‘New_Sessions’, ‘Bounce_Rate’, ‘Pages_Session’, ‘Type’], dtype=’object’)

#设置特征值X

X = np.array(traffic[[‘New_Sessions’,’Bounce_Rate’,’Pages_Session’]])

#设置目标值Y

Y = np.array(traffic[‘Type’])
#查看数据集的维度

X.shape,Y.shape

((100, 3), (100,))

X_train,X_test,y_train,y_test = cross_validation.train_test_split(X, Y, test_size=0.4, random_state=0)
#查看训练集的维度

X_train.shape, y_train.shape

((60, 3), (60,))
#查看测试集的维度

X_test.shape, y_test.shape

((40, 3), (40,))
#建立模型

clf = RandomForestClassifier()

#对模型进行训练

clf = clf.fit(X_train, y_train)

#对模型进行测试

clf.score(X_test, y_test)
clf.predict(X_test[0])
clf.predict_proba(X_test[0])