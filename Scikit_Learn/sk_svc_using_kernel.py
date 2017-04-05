from sklearn import  svm
import numpy as np
from sklearn import datasets
iris = datasets.load_iris()
iris_X = iris.data
iris_y = iris.target
np.unique(iris_y)

np.random.seed(0)
indices = np.random.permutation(len(iris_X))
print len(iris_X)

iris_X_train = iris_X[indices[:-10]]
iris_y_train = iris_y[indices[:-10]]
iris_X_test = iris_X[indices[-10:]]
iris_y_test = iris_y[indices[-10:]]

#For many estimators, including the SVMs, having datasets with unit standard deviation for each feature is important
#to get good prediction.
svc = svm.SVC(kernel='linear')
svc.fit(iris_X_train,iris_y_train)


#Using kernels
svc1 = svm.SVC(kernel='linear')
svc2 = svm.SVC(kernel='poly',degree=3)
svc3 = svm.SVC(kernel='rbf')
