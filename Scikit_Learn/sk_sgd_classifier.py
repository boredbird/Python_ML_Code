from sklearn.linear_model import SGDClassifier
X = [[0., 0.], [1., 1.]]
y = [0, 1]
clf = SGDClassifier(loss="hinge", penalty="l2")
clf.fit(X, y)
print clf.predict([[2., 2.]])
#To get the signed distance to the hyperplane use SGDClassifier.decision_function
print clf.decision_function([[2., 2.]])

clf1 = SGDClassifier(loss="log").fit(X, y)
clf1.predict_proba([[1., 1.]])
