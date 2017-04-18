from sklearn.svm import  SVC
from sklearn.multiclass import OneVsRestClassifier
from  sklearn.preprocessing import LabelBinarizer

X = [[1,2],[2,4],[4,5],[3,2],[3,1]]
y = [0,0,1,1,2]

### fit on a 1d array of multiclass labels
classif = OneVsRestClassifier(estimator=SVC(random_state=0))
print  classif.fit(X,y).predict(X)

### fit on a 2d array of binary label
y = LabelBinarizer().fit_transform(y)
print classif.fit(X,y).predict(X)

###why the result is different?????