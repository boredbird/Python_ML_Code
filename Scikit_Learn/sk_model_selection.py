#Model selection: choosing estimators and their parameters
#Score, and cross-validated scores

#every estimator exposes a score method that can judge the quality of the fit (or the prediction) on new data
from sklearn import datasets, svm
digits = datasets.load_digits()
X_digits = digits.data
y_digits = digits.target
svc = svm.SVC(C=1, kernel='linear')
svc.fit(X_digits[:-100], y_digits[:-100]).score(X_digits[-100:], y_digits[-100:])

#To get a better measure of prediction accuracy (which we can use as a proxy for goodness of fit of the model), we can
#successively split the data in folds that we use for training and testing:

import numpy as np
X_folds = np.array_split(X_digits, 3)
y_folds = np.array_split(y_digits, 3)
scores = list()
for k in range(3):
# We use 'list' to copy, in order to 'pop' later on
    X_train = list(X_folds)
    X_test = X_train.pop(k)
    X_train = np.concatenate(X_train)
    y_train = list(y_folds)
    y_test = y_train.pop(k)
    y_train = np.concatenate(y_train)
    scores.append(svc.fit(X_train, y_train).score(X_test, y_test))
print(scores)

#Cross-validation generators
#This example shows an example usage of the split method.
from sklearn.model_selection import KFold, cross_val_score
X = ["a", "a", "b", "c", "c", "c"]
k_fold = KFold(n_splits=3)
for train_indices, test_indices in k_fold.split(X):
    print('Train: %s | test: %s' % (train_indices, test_indices))


k_fold = KFold(n_splits=3)
[svc.fit(X_digits[train], y_digits[train]).score(X_digits[test], y_digits[test])
        for train, test in k_fold.split(X_digits)]

#The cross-validation score can be directly calculated using the cross_val_score helper
cross_val_score(svc, X_digits, y_digits, cv=k_fold, n_jobs=-1)
#n_jobs=-1 means that the computation will be dispatched on all the CPUs of the computer.



#Grid-search and cross-validated estimators
#Grid-search
from sklearn.model_selection import GridSearchCV, cross_val_score
Cs = np.logspace(-6, -1, 10)
clf = GridSearchCV(estimator=svc, param_grid=dict(C=Cs),n_jobs=-1)
clf.fit(X_digits[:1000], y_digits[:1000])
print clf.best_score_
print clf.best_estimator_.C
print clf.score(X_digits[1000:], y_digits[1000:])

#Cross-validated estimators
from sklearn import  linear_model,datasets
lasso = linear_model.LassoCV()
diabetes = datasets.load_diabetes()
X_diabetes = diabetes.data
y_diabetes = diabetes.target
lasso.fit(X_diabetes,y_diabetes)
print lasso.alpha_
