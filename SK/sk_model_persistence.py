from sklearn import  svm
from sklearn import  datasets
clf = svm.SVC()
iris = datasets.load_iris()
X,y = iris.data,iris.target
clf.fit(X,y)

#### save model use pickle
import  pickle
s = pickle.dumps(clf)
clf2 = pickle.loads(s)
print clf2.predict(X[0:1])
print y[0]

### save model use joblib
from sklearn.externals import  joblib
joblib.dump(clf,'sk_model_persistence.pkl')
clf3 = joblib.load('sk_model_persistence.pkl')
