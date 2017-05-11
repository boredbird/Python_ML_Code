# -*- coding:utf-8 -*-
__author__ = 'chunhui.zhang'
from sklearn import preprocessing
from sklearn.preprocessing import PolynomialFeatures
import numpy as np
X = np.array([[ 1., -1.,  2., 3],
              [ 2.,  0.,  0., 4],
              [ 0.,  1., -1., 5]])
X_scaled = preprocessing.scale(X)

scaler = preprocessing.StandardScaler().fit(X)
print scaler.mean_
print scaler.scale_
scaler.transform(X)

min_max_scaler = preprocessing.MinMaxScaler()
X_train_minmax = min_max_scaler.fit_transform(X)
X_std = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))

X_scaled = X_std / (X.max(axis=0) - X.min(axis=0)) + X.min(axis=0)


# preprocessing.robust_scale
binarizer = preprocessing.Binarizer(threshold=1.1)
binarizer.transform(X)

enc = preprocessing.OneHotEncoder()
X = [[0, 0, 3], [1, 1, 0], [0, 2, 1], [1, 0, 2]]
enc.fit(X)
enc.transform([[0, 1, 3]]).toarray()
#2,3,4


import numpy as np
from sklearn.preprocessing import Imputer
imp = Imputer(missing_values='NaN', strategy='mean', axis=0)
imp.fit([[1, 2], [np.nan, 3], [7, 6]])
Imputer(axis=0, copy=True, missing_values='NaN', strategy='mean', verbose=0)
X = [[np.nan, 2], [6, np.nan], [7, 6]]
print(imp.transform(X))


import scipy.sparse as sp
X = sp.csc_matrix([[1, 2], [0, 3], [7, 6]])
imp = Imputer(missing_values=0, strategy='mean', axis=0)
imp.fit(X)
X_test = sp.csc_matrix([[0, 2], [6, 0], [7, 6]])
print(imp.transform(X_test))


X = np.arange(6).reshape(3, 2)
poly = PolynomialFeatures(2)
poly.fit_transform(X)

import numpy as np
from sklearn.preprocessing import FunctionTransformer
transformer = FunctionTransformer(np.log1p)
X = np.array([[0, 1], [2, 3]])
transformer.transform(X)