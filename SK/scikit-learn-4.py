import numpy as ny 
from sklearn import datasets
from sklearn.cross_validation import train_test_split
from sklearn.neighbors import KNeighborsClassifier

iris = datasets.load_iris()
iris_X = iris.data
iris_Y = iris.target

# print(iris_X[:2,:])
# print(iris_Y)

X_train,X_test,Y_train,Y_test=train_test_split(iris_X,iris_Y,test_size=0.3)

knn=KNeighborsClassifier()
knn.fit(X_train,Y_train)
print(knn.predict(X_test))
print(Y_test)
